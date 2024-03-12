import streamlit as st
import pandas as pd
import numpy as np
from LEURN import LEURN
import torch
from DATA import split_and_processing
from TRAINER import Trainer
import numpy as np 
import openml 

# Streamlit application layout
st.title("LEURN")

# Initialize or reset session states if necessary
if 'init' not in st.session_state:
    st.session_state['training_completed'] = False
    st.session_state['data_chosen'] = False
    st.session_state['init'] = True
    st.session_state['selected_row']=False
    st.session_state['explanation_made']=False
    st.session_state['result']=False


# Upload csv or excel
st.subheader("File Uploader")
uploaded_file = st.file_uploader("Upload your Excel/CSV file", type=["csv", "xlsx"])
if uploaded_file is not None:
    # Reading the uploaded file
    df = pd.read_csv(uploaded_file) if uploaded_file.type == "text/csv" else pd.read_excel(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    st.subheader("Categorical Feature and Target Selection")
    # Selecting the target variable
    target = st.selectbox("Select the target variable", options=df.columns)
    
    # Define features and target
    X = df.drop(target, axis=1)
    y = df[target]
    attribute_names = X.columns
    
    
    # Select categorical variables
    st.write("Select categorical variables:")
    categoricals = [st.checkbox(f"{col} is categorical", key=col) for col in X.columns]
    
    # User input for model parameters
    st.subheader("Model Training Parameters")
    depth = st.selectbox("Select Model Depth", options=[1, 2, 3, 4, 5], index=2)
    batch_size = st.selectbox("Select Batch Size", options=[64, 128, 256, 512, 1024, 2048, 4096], index=4)
    lr = st.selectbox("Select Learning Rate", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], index=3)
    epochs = st.number_input("Enter Number of Epochs", min_value=1, max_value=1000, value=300)
    droprate = st.slider("Select Dropout Rate", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    output_type = st.radio("Select Output Type (0: regression, 1: binary classification, 2: multi-class classification)", options=[0, 1, 2], index=0)

if st.button("Train Neural Network"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Split and process
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = split_and_processing(X, y, categoricals, output_type, attribute_names)
    #Initialize model
    model = LEURN(preprocessor, depth=depth, droprate=droprate).to(device)
    #Train model
    model_trainer = Trainer(model, X_train, X_val, y_train, y_val, lr=lr, batch_size=batch_size, epochs=epochs, problem_type=output_type, verbose=False)
    model_trainer.train()
    #Load best model
    model.load_state_dict(torch.load('best_model_weights.pth'))
    #Get performances
    perf_train = model_trainer.evaluate(X_train, y_train)
    perf_val = model_trainer.evaluate(X_val, y_val)
    perf_test = model_trainer.evaluate(X_test, y_test)
    st.session_state['perf_train']=perf_train
    st.session_state['perf_val']=perf_val
    st.session_state['perf_test']=perf_test

    #Save test dataset and model to explain/generate later
    X_test_inverse = preprocessor.inverse_transform_X(X_test)
    X_test_inverse.to_csv('test.csv',index=False)
    st.session_state['training_completed'] = True
    st.session_state['model'] = model  # Adjusted for compatibility



if st.session_state['training_completed'] == True:  

    #Print performances
    st.write("Here are performances, try different hyperparameters if not satisfied")
    if output_type == 0:
        st.subheader("Training Results (MSE)")
    elif output_type == 1:
        st.subheader("Training Results (ROC-AUC)")
    else:
        st.subheader("Training Results (ACC)")

    st.write(f"Training Score: {st.session_state['perf_train']:.4f}")
    st.write(f"Validation Score: {st.session_state['perf_val']:.4f}")
    st.write(f"Test Score: {st.session_state['perf_test']:.4f}")
    
    
    # File uploader for explanation

    st.subheader("Explain New Inputs")
    uploaded_file_to_explain = st.file_uploader("Upload your Excel/CSV file to explain", type=["csv", "xlsx"])
    print(uploaded_file_to_explain)
    if uploaded_file_to_explain is not None:
        # Reading the uploaded file
        
        X_test_inverse = pd.read_csv(uploaded_file_to_explain) if uploaded_file_to_explain.type == "text/csv" else pd.read_excel(uploaded_file_to_explain)
        
        # Save DataFrame 
        st.session_state['X_test_inverse_df'] = X_test_inverse.to_json()
        st.session_state['data_chosen'] = True  # Flag to indicate data is chosen
        
    
    if st.session_state['data_chosen'] == True:
        # Load DataFrame from session state
        X_test_inverse = pd.read_json(st.session_state['X_test_inverse_df'])
        
        # Always display the DataFrame to ensure it's visible for selection
        st.write("Test DataFrame:")
        st.write(X_test_inverse)
        
        # Let users select a row, selection is dynamic and updates session state
        selected_index = st.selectbox("Select a row:", options=X_test_inverse.index, key="selected_index")

        selected_row = X_test_inverse.loc[[st.session_state['selected_index']]]
        st.write("Selected Data for Explanation:")
        st.write(selected_row)
        st.session_state['selected_row'] = selected_row
        
        #Explain selected row
        if st.button("Explain"):
            model=st.session_state['model']
            Exp_df_test_sample,result,result_original_format=model.explain(torch.from_numpy(model.preprocessor.transform_X(st.session_state['selected_row']).values.astype('float32')),include_causal_analysis=True)
            st.session_state['explanation_made']=True
            st.session_state['Exp_df_test_sample']=Exp_df_test_sample
            st.session_state['result_original_format']=result_original_format
            st.session_state['result']=result
            
        #Print explanations
        if st.session_state['explanation_made']==True:
            st.write("Explanation DataFrame:")
            st.write(st.session_state['Exp_df_test_sample'])
            st.write("Predicted Output: (Network format)")
            st.write(st.session_state['result'].detach().numpy().astype('str'))
            if output_type==1:
                if np.sign(st.session_state['result'].detach().numpy())>0:
                    st.write("Result here is positive; this means output class below is represented by positive sign. In the explanation dataframe, positive contributions increase class likelihood")
                else:
                    st.write("Result here is negative; this means output class below is represented by negative sign. In the explanation dataframe, negative contributions increase class likelihood")
    
            st.write("Predicted Output: (original format)")
            st.write(st.session_state['result_original_format'])
    
    #Data generation part
    st.subheader("Generate Data From Scratch")
    if st.button("Generate"):
        model=st.session_state['model']
        generated_sample_nn_friendly, generated_sample_original_input_format,output=model.generate()
        Exp_df_generated_sample,result,result_original_format=model.explain(generated_sample_nn_friendly,include_causal_analysis=True)
        st.write("Explanation DataFrame:")
        st.write(Exp_df_generated_sample)
        st.write("Predicted Output: (Network format)")
        st.write(result.detach().numpy().astype('str'))
        if output_type==1:
            if np.sign(result.detach().numpy())>0:
                st.write("Result here is positive; this means output class below is represented by positive sign. In the explanation dataframe, positive contributions increase class likelihood")
            else:
                st.write("Result here is negative; this means output class below is represented by negative sign. In the explanation dataframe, negative contributions increase class likelihood")

        st.write("Predicted Output: (original format)")
        st.write(result_original_format)
        
            

            
        
