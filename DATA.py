"""
@author: Caglar Aytekin
contact: caglar@deepcause.ai 
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
pd.set_option('display.max_rows', None)  # None means show all rows
pd.set_option('display.max_columns', None)  # None means show all columns
pd.set_option('display.width', None)  # Use appropriate width to display columns
pd.set_option('display.max_colwidth', None)  # Show full content of each column

warnings.filterwarnings("ignore")

def split_and_processing(X,y,categoricals,output_type,attribute_names):
    #If every entryin a column  of a dataframe is None drop it
    columns_to_keep_mask = ~X.isna().all()
    X = X.dropna(axis=1, how='all') 
    # Update the categoricals list to reflect the columns not dropped
    categoricals = [cat for cat, keep in zip(categoricals, columns_to_keep_mask) if keep]
    attribute_names= [cat for cat, keep in zip(attribute_names, columns_to_keep_mask) if keep]
    
    
        
    # Split into train and remaining
    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split remaining into validation and test
    X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)

    # Initialize preprocessor
    preprocessor=DataProcessor(categoricals,output_type)

    #Fit and transform for training set
    X_train=torch.from_numpy(preprocessor.fit_transform_X(X_train).values).float()
    y_train=torch.from_numpy(preprocessor.fit_transform_y(y_train)).float()
    if output_type<2:
        y_train=y_train.unsqueeze(dim=-1)
    else:
        y_train=y_train.long()

    #Transform for validation and test set
    X_val=torch.from_numpy(preprocessor.transform_X(X_val).values).float()
    y_val=torch.from_numpy(preprocessor.transform_y(y_val)).float()
    if output_type<2:
        y_val=y_val.unsqueeze(dim=-1)
    else:
        y_val=y_val.long()

    X_test=torch.from_numpy(preprocessor.transform_X(X_test).values).float()
    y_test=torch.from_numpy(preprocessor.transform_y(y_test)).float()
    if output_type<2:
        y_test=y_test.unsqueeze(dim=-1)
    else:
        y_test=y_test.long()
        
    preprocessor.attribute_names=attribute_names
    preprocessor.output_type=output_type
    
    #Determine class no
    if output_type==0:
        output_dim=y_train.shape[1]
    elif output_type==1:
        output_dim=1
    else:
        output_dim=len(np.unique(y_train))
        
    preprocessor.output_dim=output_dim   
    return X_train,X_val,X_test,y_train,y_val,y_test,preprocessor



class DataProcessor:
    def __init__(self, categoricals, output_type):
        self.categoricals = categoricals
        self.output_type = output_type
        self.label_encoders = {}
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.most_common_categories = {}
        self.target_encoder = None # For binary and multiclass
        self.unique_targets = None # To store unique targets for binary classification
        self.category_details=[]
        self.suggested_embeddings=None
        self.encoders_for_nn={}
    
    def fit_transform_X(self, X):
        

        # Convert all numerical columns to float precision
        X.iloc[:, ~np.array(self.categoricals)] = X.iloc[:, ~np.array(self.categoricals)].astype(float)
        X.iloc[:, np.array(self.categoricals)] = X.iloc[:, np.array(self.categoricals)].astype(str)

        X_transformed = X.copy()
        for i, is_categorical in enumerate(self.categoricals):
            if is_categorical:
                encoder = LabelEncoder()
                X_transformed.iloc[:, i] = encoder.fit_transform(X.iloc[:, i])
                self.label_encoders[i] = encoder
                self.encoders_for_nn[X_transformed.columns[i]] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                self.most_common_categories[i] = X.iloc[:, i].mode()[0]
                self.category_details.append((i, len(encoder.classes_)))
            else:
                # Fill missing values with the median for numerical columns
                X_transformed.iloc[:, i] = X.iloc[:, i].fillna(X.iloc[:, i].median())
                
        # Scale numerical features
        numerical_features = X_transformed.iloc[:, ~np.array(self.categoricals)]
        if numerical_features.shape[-1]>0:
            self.scaler.fit(numerical_features)
            X_transformed.iloc[:, ~np.array(self.categoricals)] = self.scaler.transform(numerical_features)
        self.suggested_embeddings=[max(2, int(np.log2(x[1]))) for x in self.category_details]
        
        return X_transformed.astype(float)
    
    def transform_X(self, X):
        X.iloc[:, np.array(self.categoricals)] = X.iloc[:, np.array(self.categoricals)].astype(str)
        X_transformed = X.copy()
        for i, is_categorical in enumerate(self.categoricals):
            if is_categorical:
                encoder = self.label_encoders[i]
                # Transform categories, replace unseen with most common category
                X_transformed.iloc[:, i] = X.iloc[:, i].map(lambda x: x if x in encoder.classes_ else self.most_common_categories[i])
                X_transformed.iloc[:, i] = encoder.transform(X_transformed.iloc[:, i])
            else:
                X_transformed.iloc[:, i] = X.iloc[:, i].fillna(X.iloc[:, i].mean())
                
        # Scale numerical features
        numerical_features = X_transformed.iloc[:, ~np.array(self.categoricals)]
        if numerical_features.shape[-1]>0:
            X_transformed.iloc[:, ~np.array(self.categoricals)] = self.scaler.transform(numerical_features)
        
        return X_transformed.astype(float)
    
    
    def inverse_transform_X(self, sample):
        #inverse transform from pytorch tensor
        sample=sample.detach().numpy()
        sample_inverse_transformed = pd.DataFrame(sample.copy())
        
        #Handle numerical features
        numerical_features_indices = np.where(~np.array(self.categoricals))[0]
        if len(numerical_features_indices)>0:
            sample_inverse_transformed.iloc[:,numerical_features_indices] = self.scaler.inverse_transform(sample[:,numerical_features_indices])
        

        for i, is_categorical in enumerate(self.categoricals):
            if is_categorical:
                encoder = self.label_encoders[i]
                sample_inverse_transformed.iloc[:, i] = encoder.inverse_transform(sample[:, i].astype('int'))
        sample_inverse_transformed.columns = self.attribute_names
        return sample_inverse_transformed

    
    def fit_transform_y(self, y):
        if self.output_type == 0: # Regression
            y_transformed = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        elif self.output_type == 1: # Binary classification
            self.unique_targets = y.unique()
            mapping = {category: idx for idx, category in enumerate(self.unique_targets)}
            y_transformed = y.map(mapping).astype(int).values
        elif self.output_type == 2: # Multiclass classification
            self.target_encoder = LabelEncoder()
            y_transformed = self.target_encoder.fit_transform(y)
        else:
            raise ValueError("Invalid output type")
        return y_transformed
    
    def transform_y(self, y):
        if self.output_type == 0: # Regression
            y_transformed = self.target_scaler.transform(y.values.reshape(-1, 1)).flatten()
        elif self.output_type == 1: # Binary classification
            mapping = {category: idx for idx, category in enumerate(self.unique_targets)}
            y_transformed = y.map(mapping).astype(int).values
        elif self.output_type == 2: # Multiclass classification
            y_transformed = self.target_encoder.transform(y)
        else:
            raise ValueError("Invalid output type")
        return y_transformed
    
    def inverse_transform_y(self, nn_output):
        if self.output_type == 0: # Regression
            y_transformed=nn_output.squeeze().detach().numpy()
            return self.target_scaler.inverse_transform(y_transformed.reshape(-1, 1)).flatten()
        elif self.output_type == 1: # Binary classification
            y_transformed=int(np.round(torch.sigmoid(nn_output).squeeze().detach().numpy()))
            inverse_mapping = {idx: category for idx, category in enumerate(self.unique_targets)}
            return inverse_mapping[y_transformed]
        elif self.output_type == 2: # Multiclass classification
            y_transformed=int(np.round(torch.argmax(nn_output).squeeze().detach().numpy()))
            return self.target_encoder.inverse_transform([y_transformed])
        else:
            raise ValueError("Invalid output type")
        
