"""
@author: Caglar Aytekin
contact: caglar@deepcause.ai 
"""
# %% IMPORT
from LEURN import LEURN
import torch
from DATA import split_and_processing
from TRAINER import Trainer
import numpy as np 
import openml 



#DEMO FOR CREDIT SCORING DATASET: OPENML ID : 31
#MORE INFO: https://www.openml.org/search?type=data&sort=runs&id=31&status=active
#%% Set Neural Network Hyperparameters
depth=2
batch_size=1024
lr=1e-3
epochs=500
droprate=0.0
output_type=1 #0: regression, 1: binary classification, 2: multi-class classification

#%%  Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


#%%  Load the dataset
#Read dataset from openml
open_ml_dataset_id=31
dataset = openml.datasets.get_dataset(open_ml_dataset_id)
X, y, categoricals, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
#Alternatively load your own dataset from another source (excel,csv etc)
#Be mindful that X and y should be dataframes, categoricals is a boolean list indicating categorical features, attribute_names is a list of feature names

# %% Process data, save useful statistics
X_train,X_val,X_test,y_train,y_val,y_test,preprocessor=split_and_processing(X,y,categoricals,output_type,attribute_names)



#%% Initialize model, loss function, optimizer, and learning rate scheduler
model = LEURN(preprocessor, depth=depth,droprate=droprate).to(device)


#%%Train model
model_trainer=Trainer(model, X_train, X_val, y_train, y_val,lr=lr,batch_size=batch_size,epochs=epochs,problem_type=output_type)
model_trainer.train()
#Load best weights
model.load_state_dict(torch.load('best_model_weights.pth'))

#%%Evaluate performance
perf=model_trainer.evaluate(X_train, y_train)
perf=model_trainer.evaluate(X_test, y_test)
perf=model_trainer.evaluate(X_val, y_val)

#%%TESTS
model.eval()

#%%Check sample in original format:
print(preprocessor.inverse_transform_X(X_test[0:1]))
#%% Explain single example
Exp_df_test_sample,result,result_original_format=model.explain(X_test[0:1])
#%%  Check results
print(result,result_original_format)
#%% Check explanation
print(Exp_df_test_sample)
#%% tests
#model output and sum of contributions should be the same
print(result,model.output,model(X_test[0:1]),Exp_df_test_sample['Contribution'].values.sum())


#%% GENERATION FROM SAME CATEGORY
generated_sample_nn_friendly, generated_sample_original_input_format,output=model.generate_from_same_category(X_test[0:1])
#%%Check sample in original format:
print(preprocessor.inverse_transform_X(X_test[0:1]))
print(generated_sample_original_input_format)
#%% Explain single example
Exp_df_generated_sample,result,result_original_format=model.explain(generated_sample_nn_friendly)
print(Exp_df_generated_sample)
print(Exp_df_test_sample.equals(Exp_df_generated_sample)) #this should be true


#%% GENERATE FROM SCRATCH
generated_sample_nn_friendly, generated_sample_original_input_format,output=model.generate()
Exp_df_generated_sample,result,result_original_format=model.explain(generated_sample_nn_friendly)
print(Exp_df_generated_sample)
print(result,result_original_format)



    
    
