"""
@author: Caglar Aytekin
contact: caglar@deepcause.ai 
"""
import torch
import torch.nn as nn
import random 
import numpy as np 
import pandas as pd
import copy
class CustomEncodingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tau,alpha):
        ctx.save_for_backward(x, tau)
        # Perform the tanh operation on (x + tau) 
        y = torch.tanh(x + tau)
        # The actual forward output : binarized  output
        forward_output = alpha * (2 * torch.round((y + 1) / 2) - 1) + (1-alpha)*y
        return forward_output

    @staticmethod
    def backward(ctx, grad_output):
        x, tau = ctx.saved_tensors
        # Use the derivative of tanh for the backward pass: 1 - tanh^2(x + tau)
        grad_input = grad_output * (1 - torch.tanh(x + tau) ** 2)
        return grad_input, grad_input,None  # Assuming tau also requires gradient

# Wrapping the custom function in a nn.Module for easier use
class EncodingLayer(nn.Module):
    def __init__(self):
        super(EncodingLayer, self).__init__()
    def forward(self, x, tau,alpha):
        return CustomEncodingFunction.apply(x, tau,alpha)
    
class LEURN(nn.Module):
    def __init__(self, preprocessor,depth,droprate):
        """
        Initializes the model.
        
        Parameters:
        - preprocessor: A class containing useful info about the dataset 
            - Including: attribute names, categorical features details, suggested embedding size for each category, output type, output dimension, transformation information
        - depth: Depth of the network
        - droprate: dropout rate
        """
        super(LEURN, self).__init__()
        
        #Find categorical indices and category numbers for each
        self.alpha=1.0
        self.preprocessor=preprocessor
        self.attribute_names=preprocessor.attribute_names
        self.label_encoders=preprocessor.encoders_for_nn
        self.categorical_indices = [info[0] for info in preprocessor.category_details]
        self.num_categories = [info[1] for info in preprocessor.category_details]

        #If embedding_size is integer, cast it to all categories
        if isinstance(preprocessor.suggested_embeddings, int):
            embedding_sizes = [preprocessor.suggested_embeddings] * len(self.categorical_indices)
        else:
            assert len(preprocessor.suggested_embeddings) == len(self.categorical_indices), "Length of embedding_size must match number of categorical features"
            embedding_sizes = preprocessor.suggested_embeddings
        
        self.embedding_sizes=embedding_sizes
        
        #Embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim) 
            for num_categories, embedding_dim in zip(self.num_categories, embedding_sizes)
        ])
        
        self.total_embedding_size = sum(embedding_sizes) #number of categorical features for NN
        self.non_cat_input_dim = len(self.attribute_names) - len(self.categorical_indices) #Number of numerical features for NN
        self.nn_input_dim = self.total_embedding_size + self.non_cat_input_dim #Number of features for NN
        

        #LAYERS
        
        self.tau_initial = nn.Parameter(torch.zeros(1,self.nn_input_dim))  # Initial tau as a learnable parameter
        self.layers = nn.ModuleList()
        self.depth = depth
        self.output_type=preprocessor.output_type
        
        for d_now in range(depth):
            # Each iteration adds an encoding layer followed by a dropout and then a linear layer
            self.layers.append(EncodingLayer())
            self.layers.append(nn.Dropout1d(droprate))
            linear_layer = nn.Linear((d_now + 1) * self.nn_input_dim, self.nn_input_dim)
            self._init_weights(linear_layer, (d_now + 1) * self.nn_input_dim) #special layer initialization
            self.layers.append(linear_layer)
        
        
        # Final stage: dropout and linear layer
        self.final_dropout=nn.Dropout1d(droprate)
        self.final_linear = nn.Linear(depth * self.nn_input_dim, self.preprocessor.output_dim)
        self._init_weights(self.final_linear, depth * self.nn_input_dim)

    def set_alpha(self, alpha):
        """Method to update the dynamic parameter."""
        self.alpha = alpha

    def _init_weights(self, layer, input_dim):
        # Custom initialization 
        # Considering the binary (-1,1) nature of the input, 
        # when we initialize layer in (-1/dim,1/dim) range, output is bounded at (-1,1)
        # Knowing our input is roughly at (-1,1) range, this serves as good initialization for tau
        stdv = 1. / torch.tensor(input_dim)
        layer.weight.data.uniform_(-stdv, stdv)

    
    def forward(self, x):
        # Defines forward function for provided input: Normalizes numericals, embeds categoricals, and gives to neural network.

                    
        # Separate categorical and numerical features for easier handling   
        cat_features = [x[:, i].long() for i in self.categorical_indices]
        non_cat_features = [x[:, i] for i in range(x.size(1)) if i not in self.categorical_indices]
        non_cat_features = torch.stack(non_cat_features, dim=1) if non_cat_features else x.new_empty(x.size(0), 0)
        
        # Embed categoricals
        embedded_features = [embedding(cat_feature) for embedding, cat_feature in zip(self.embeddings, cat_features)]
        # Combine categoricals and numericals
        try:
            embedded_features = torch.cat(embedded_features, dim=1)
            nninput = torch.cat([embedded_features, non_cat_features], dim=1)
        except:
            nninput=non_cat_features
        
        self.nninput=nninput
        
        # Forward pass neural network
        output=self.forward_from_embeddings(self.nninput)
        self.output=output
        return output

    def forward_from_embeddings(self,x):
        # Forward function for normalized numericals and embedded categoricals
        tau=self.tau_initial 
        tau=torch.repeat_interleave(tau,x.shape[0],0)  #tau is 1xF, cast it for batch
        # For each depth
        for i in range(0, self.depth * 3, 3):
            # encode, drop and find next tau
            encoding_layer = self.layers[i]
            dropout_layer = self.layers[i + 1]
            linear_layer = self.layers[i + 2]
            #encode and drop
            encoded_x =dropout_layer( encoding_layer(x, tau,self.alpha))
            #save encodings and thresholds
            #notice that threshold is -tau, not tau since we binarize x+tau
            if i==0:
                encodings=encoded_x
                taus=-tau
            else:
                encodings=torch.cat((encodings,encoded_x),dim=-1)
                taus=torch.cat((taus,-tau),dim=-1)
            #find next thresholds
            tau = linear_layer(encodings) #not used, redundant for last layer
        
        self.encodings=encodings
        self.taus=taus
        #Final layer: drop and linear
        output=self.final_linear(self.final_dropout(encodings))

        return output
    
    
    def find_boundaries(self, x):
        """
        Given input, find boundaries for numerical features and valid categories for categorical features
        Can accept unnormalized and not embedded input - set embedding False
        """
        # Ensure x is the correct shape [1, input_dim]
        if x.ndim == 1:
            x = x.unsqueeze(0)  # Add batch dimension if not present
        
        # Perform a forward pass to update self.encodings and self.taus
        # to update self.taus

        self(x)
        
        # self.taus has the shape [1, depth * input_dim]
        # reshape to [depth, input_dim] for easier boundary finding
        taus_reshaped = self.taus.view(self.depth, self.nn_input_dim) 
        
        # embedded and normalized input
        embedded_x=self.nninput
        
        # Initialize boundaries - numericals are in (-1,1) range and categoricals are from embeddings.
        # So -100,100 is safe min and max. -inf,+inf is not chosen since problematic for later sampling
        upper_boundaries = torch.full((embedded_x.size(1),), 100.0)
        lower_boundaries = torch.full((embedded_x.size(1),), -100.0)
        
        # Compare each threshold in self.taus with the corresponding input value
        for feature_index in range(self.nn_input_dim):
            for depth_index in range(self.depth):
                threshold = taus_reshaped[depth_index, feature_index]
                input_value = embedded_x[0, feature_index]
                
                # If the threshold is greater than the input value and less than the current upper boundary, update the upper boundary
                if threshold > input_value and threshold < upper_boundaries[feature_index]:
                    upper_boundaries[feature_index] = threshold
                
                # If the threshold is less than the input value and greater than the current lower boundary, update the lower boundary
                if threshold < input_value and threshold > lower_boundaries[feature_index]:
                    lower_boundaries[feature_index] = threshold
        
        # Convert boundaries to a list of tuples [(lower, upper), ...] for each feature
        boundaries = list(zip(lower_boundaries.tolist(), upper_boundaries.tolist()))
        
        
        self.upper_boundaries=upper_boundaries
        self.lower_boundaries=lower_boundaries
        

        return boundaries
    
    def categories_within_boundaries(self):
        """
        For each categorical feature, checks if embedding weights fall within the specified upper and lower boundaries.
        Returns a dictionary with categorical feature indices as keys and lists of category indices that fall within the boundaries.
        """
        categories_within_bounds = {}
        emb_st=0
        for cat_index, emb_layer in zip(range(len(self.categorical_indices)), self.embeddings):
            # Extract upper and lower boundaries for this categorical feature
            lower_bound=self.lower_boundaries[emb_st:emb_st+self.embedding_sizes[cat_index]]
            upper_bound=self.upper_boundaries[emb_st:emb_st+self.embedding_sizes[cat_index]]
            emb_st=emb_st+self.embedding_sizes[cat_index]
            # Initialize list to hold categories that fall within boundaries
            categories_within = []
    
            # Iterate over each embedding vector in the layer
            for i, weight in enumerate(emb_layer.weight):
                # Check if the embedding weight falls within the boundaries
                if torch.all(weight >= lower_bound) and torch.all(weight <= upper_bound):
                    categories_within.append(i)  # Using index i as category identifier
            
            # Store the categories that fall within the boundaries for this feature
            categories_within_bounds[cat_index] = categories_within
    
        return categories_within_bounds
    
    def explain(self,x):
        """
        Explains decisions of the neural network for input sample.
        For numericals, extracts upper and lower boundaries on the sample
        For categoricals displays possible categories
        Also calculates contributions of each feature to final result
        """
        self.find_boundaries(x) #find upper, lower boundaries for all nn inputs
        
        #find valid categories for categorical features
        valid_categories=self.categories_within_boundaries()
        
        #numerical boundaries
        upper_numerical=self.upper_boundaries[sum(self.embedding_sizes):].detach().numpy()
        lower_numerical=self.lower_boundaries[sum(self.embedding_sizes):].detach().numpy()
        
        #Find contribution from each feature in final linear layer, distribute bias evenly
        contributions=self.encodings * self.final_linear.weight + self.final_linear.bias.unsqueeze(dim=-1)/self.final_linear.weight.shape[1]
        contributions=contributions.detach().resize_((contributions.shape[0], contributions.shape[1]//self.nn_input_dim,self.nn_input_dim))
        contributions=torch.sum(contributions,dim=1)
        
        # Initialize an empty list to store the summed contributions
        summed_contributions = []
        
        # Initialize start index for slicing
        start_idx = 0
        
        #Sum contribution of each categorical within respective embedding
        for size in self.embedding_sizes:
            # Calculate end index for the current chunk
            end_idx = start_idx + size
            
            # Sum the contributions in the current chunk
            chunk_sum = contributions[:, start_idx:end_idx].sum(dim=1, keepdim=True)
            
            # Append the summed chunk to the list
            summed_contributions.append(chunk_sum)
            
            # Update the start index for the next chunk
            start_idx = end_idx
        
        # If there are remaining elements not covered by embedding_sizes, add them as is (numerical features)
        if start_idx < contributions.shape[1]:
            remaining = contributions[:, start_idx:]
            summed_contributions.append(remaining)
        
        # Concatenate the summed contributions back into a tensor
        summed_contributions = torch.cat(summed_contributions, dim=1)
        # This is to handle multi-class explanations, for binary this is 0 automatically
        # Note: multi-output regression is not supported yet. This will just return largest regressed value's explanations
        highest_index=torch.argmax(summed_contributions.sum(dim=1))
        # This is contribution from each feature
        result=summed_contributions[highest_index]
        self.result=result

        #Explanation and Contribution formats are in ordered format (categoricals first, numericals later)
        #Bring them to original format in user input
        #Combine categoricals and numericals explanations and contributions
        Explanation = [None] * (len(self.categorical_indices) + len(upper_numerical))
        Contribution = np.zeros((len(self.categorical_indices) + len(upper_numerical),))   
        
        # Fill in the categorical samples
        for j, cat_index in enumerate(self.categorical_indices):
            Explanation[cat_index] = valid_categories[j]
            Contribution[cat_index] = result[j].numpy()
        
        
        #INVERSE TRANSFORM PART 1-------------------------------------------------------------------------------------------
        #Inverse transform upper and lower_numericals
        len_num=len(upper_numerical)
        if len_num>0:
            upper_numerical=self.preprocessor.scaler.inverse_transform(upper_numerical.reshape(1,-1))
            lower_numerical=self.preprocessor.scaler.inverse_transform(lower_numerical.reshape(1,-1))
            if len_num>1:
                upper_numerical=np.squeeze(upper_numerical)
                lower_numerical=np.squeeze(lower_numerical)
            upper_iter = iter(upper_numerical)
            lower_iter = iter(lower_numerical)
        
        
        cnt=0
        for i in range(len(Explanation)):
            if Explanation[i] is None:
                #Note the denormalization here
                Explanation[i] = next(lower_iter),next(upper_iter)
                if len(self.categorical_indices)>0:
                    Contribution[i] = result[j+cnt+1].numpy()
                else:
                    Contribution[i] = result[cnt].numpy()
                cnt=cnt+1

        attribute_names_list = []
        revised_explanations_list = []
        contributions_list = []
        # Process each feature to fill lists

        for idx, attr_name in enumerate(self.attribute_names):
            if isinstance(Explanation[idx], list):  # Categorical
                #INVERSE TRANSFORM PART 2-------------------------------------------------------------------------------------------
                #Inverse transform categoricals
                category_names = [key for key, value in self.label_encoders[attr_name].items() if value in Explanation[idx]]
                revised_explanation = " ,OR, ".join(category_names)
            elif isinstance(Explanation[idx], tuple):  # Numerical
                revised_explanation = f"{Explanation[idx][0].item()} to {Explanation[idx][1].item()}"
            else:
                revised_explanation = "Unknown"  #shouldn't really happen

            # Append to lists
            attribute_names_list.append(attr_name)
            revised_explanations_list.append(revised_explanation)
            contributions_list.append(Contribution[idx] if idx < len(Contribution) else None)

        # Construct DataFrame
        Explanation_df = pd.DataFrame({
            'Name': attribute_names_list,
            'Category': revised_explanations_list,
            'Contribution': contributions_list
        })
        
        

        
        result=self.preprocessor.inverse_transform_y(self.output)
        # Explanation_df['Result'] = [result] * len(Explanation_df)

        return Explanation_df,self.output,result
    
    
    def sample_from_boundaries(self):
        """
        Assumes higher and lower boundaries are already extracted (eg self.explain is run on one input already)
        Samples a value for each feature within the specified upper and lower boundaries stored in the class instance.
        For numericals, samples a value, for categoricals samples a category from possible categories
        Returns:
        - A tensor containing sampled values within the given boundaries for each feature.
        """
        #First sample from categories
        categories_within_bounds=self.categories_within_boundaries()
        try:
            sampled_indices = [random.choice(categories) for categories in categories_within_bounds.values()]
        except:
            categories_within_bounds=self.categories_within_boundaries()
        
        #Then from numericals
        samples = []
        cnt=0
        for lower, upper in zip(self.lower_boundaries[sum(self.embedding_sizes):], self.upper_boundaries[sum(self.embedding_sizes):]):
            # Sample from a uniform distribution between lower and upper boundaries
            sample = lower + (upper - lower) * torch.rand(1)
            samples.append(sample)
            cnt=cnt+1
        

        #Combine categoricals and numericals
        # Initialize an empty list to hold the combined samples
        combined_samples = [None] * (len(self.categorical_indices) + len(samples))
        
        # Fill in the categorical samples
        for i, cat_index in enumerate(self.categorical_indices):
            combined_samples[cat_index] = torch.tensor([sampled_indices[i]], dtype=torch.float)
        
        # Fill in the numerical samples
        num_samples_iter = iter(samples)
        for i in range(len(combined_samples)):
            if combined_samples[i] is None:
                combined_samples[i] = next(num_samples_iter)
        
        # Combine into a single tensor
        combined_tensor = torch.cat(combined_samples, dim=-1)
        return combined_tensor.unsqueeze(dim=0)
    
    
    def generate(self):
        """
        Generates a data sample from learned network
        """
        def sample_with_tau(tau,max_bound,min_bound):
            # Sample according to tau, lower and upper bounds
            sampled=torch.zeros((self.nn_input_dim))
            st=0
            # Randomly pick from valid categories
            for embedding in self.embeddings:
                categories_within = []
                
                # Iterate over each embedding vector in the layer
                for i, weight in enumerate(embedding.weight):
                    # Check if the embedding weight falls within the boundaries
                    if torch.all(weight >= min_bound[st:st+embedding.weight.shape[-1]]) and torch.all(weight <= max_bound[st:st+embedding.weight.shape[-1]]):
                        categories_within.append(i)  # Using index i as category identifier
                feature_now=embedding.weight[np.random.choice(categories_within),:]
                cnt=0
                for j in range(st,st+embedding.weight.shape[-1]):
                    if feature_now[cnt]>-tau[0,j]:
                        sampled[j]=1.0
                    elif feature_now[cnt]<=-tau[0,j]:
                        sampled[j]=-1.0
                    cnt=cnt+1
                st=st+embedding.weight.shape[-1]
                        
            #Randomly sample for numericals
            for i in range(st,self.nn_input_dim):
                if -tau[0,i]>max_bound[i]: #In this case you have to pick -1
                    sampled[i]=-1.0
                elif -tau[0,i]<=min_bound[i]: #In this case you have to pick 1
                    sampled[i]=1.0
                else:
                    sampled[i] = (torch.randint(low=0, high=2, size=(1,)) * 2 - 1).float()
            return sampled
        
        def bound_update(tau,max_bound,min_bound,sampled):
            for i in range(self.nn_input_dim):
                if sampled[i]>0: #means input is larger than -tau, so -tau might set a lower bound
                    if -tau[0,i]>min_bound[i]:
                        min_bound[i]=-tau[0,i]
                elif sampled[i]<=0: #means input is smaller than -tau, so -tau might set an upper bound
                    if -tau[0,i]<max_bound[i]:
                        max_bound[i]=-tau[0,i]
            return max_bound,min_bound
        
        # Read first tau
        tau=self.tau_initial

        # Set initial maximum and minimum bounds
        max_bound=torch.zeros((self.nn_input_dim))+100.0
        min_bound=torch.zeros((self.nn_input_dim))-100.0

        
        for i in range(0, self.depth * 3, 3):
            encoding_layer = self.layers[i] #NOT USED HERE, WE ENCODE RANDOMLY MANUALLY
            dropout_layer = self.layers[i + 1]
            linear_layer = self.layers[i + 2]
            #Sample with current tau
            sample_now=sample_with_tau(tau,max_bound,min_bound)
            #Update bounds with new sample
            max_bound,min_bound=bound_update(tau,max_bound,min_bound,sample_now)
            encoded_x = dropout_layer(sample_now.unsqueeze(dim=0))
            if i==0:
                encodings=encoded_x
                taus=-tau
            else:
                encodings=torch.cat((encodings,encoded_x),dim=-1)
                taus=torch.cat((taus,-tau),dim=-1)

            tau = linear_layer(encodings) #not used for last layer
            
        
        self.encodings=encodings
        self.taus=taus
        self.upper_boundaries=torch.clone(max_bound)
        self.lower_boundaries=torch.clone(min_bound)
        
        generated_sample=self.sample_from_boundaries()
        ##Check if manually found and network generated boundaries are same
        # if torch.equal(self.upper_boundaries,max_bound) and torch.equal(self.lower_boundaries,min_bound):
        #     print(True)
        
        self.explain(generated_sample)
        generated_sample_original_format=self.preprocessor.inverse_transform_X(generated_sample)
        result=self.preprocessor.inverse_transform_y(self.output)
        
        return generated_sample,generated_sample_original_format,result
    
    def generate_from_same_category(self,x):
        self.explain(x)
        generated_sample=self.sample_from_boundaries()
        generated_sample_original_format=self.preprocessor.inverse_transform_X(generated_sample)
        result=self.preprocessor.inverse_transform_y(self.output)
        return generated_sample,generated_sample_original_format,result
    

