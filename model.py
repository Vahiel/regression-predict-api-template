"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json
import seaborn as sns
from datetime import datetime
import math
from scipy.stats import pearsonr
from statsmodels.graphics.correlation import plot_corr
import statsmodels.formula.api as sm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import OLSInfluence as influence
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    
    df_riders = pd.read_csv('/Users/Vahiel/regression-predict-api-template/utils/data/riders.csv', index_col=0)
    df_train = pd.read_csv('/Users/Vahiel/regression-predict-api-template/utils/data/train_data.csv', index_col=0)
    df_test = pd.read_csv('/Users/Vahiel/regression-predict-api-template/utils/data/test_data.csv', index_col=0)

    df = pd.merge(left=df_train, right=df_riders, how='left',
                      left_on='Rider Id', right_on='Rider Id').set_index(df_train.index)
    df_t = pd.merge(left=df_test, right=df_riders,
                        how='left', left_on='Rider Id', right_on='Rider Id').set_index(df_test.index)

    df.columns = df.columns.str.replace('-','').str.replace(' ','_').str.replace('_\(.*\)','')
    df_t.columns = df_t.columns.str.replace('-','').str.replace(' ','_').str.replace('_\(.*\)','')

    df = df.drop(['User_Id','Vehicle_Type','Rider_Id','Arrival_at_Destination__Day_of_Month',
                  'Arrival_at_Destination__Weekday','Arrival_at_Destination__Time'], axis=1)

    df_t = df_t.drop(['User_Id','Vehicle_Type','Rider_Id'], axis=1)

    df = df.drop('Precipitation_in_millimeters', axis=1)
    df_t = df_t.drop('Precipitation_in_millimeters', axis=1)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
    imputer.fit(df[['Temperature']])  
    df['Temperature'] = imputer.transform(df[['Temperature']]).round(1)
    df_t['Temperature'] = imputer.transform(df_t[['Temperature']]).round(1)

    df = df.drop(['Pickup_Lat','Pickup_Long','Destination_Lat','Destination_Long'], axis=1)
    df_t = df_t.drop(['Pickup_Lat','Pickup_Long','Destination_Lat','Destination_Long'], axis=1)

    def time_to_seconds(data):
            ''' Docstring'''

            time_list = [col for col in data.columns if ('Time' in col and col!='Time_from_Pickup_to_Arrival')]

            # Change columns to datetime
            time_cols = pd.DataFrame(list(map(lambda x : [datetime.strptime(t, '%I:%M:%S %p').time() 
                                                          for t in data[x]], time_list))).T

            # Change columns to be represented in total seconds
            sec_cols = pd.DataFrame(list(map(lambda x : [((t.hour*60 + t.minute)*60 + t.second) 
                                                         for t in time_cols[x]],    time_cols.columns))).T
            sec_cols.columns = time_list
            data[time_list] = sec_cols.set_index(data.index)

    time_to_seconds(df)
    time_to_seconds(df_t)

    # Train set
    df = pd.get_dummies(df, drop_first=True, prefix='Business_or')
    # Test set
    df_t = pd.get_dummies(df_t, drop_first=True, prefix='Business_or')
    
    def yx_split(data):
        ''' Docstring'''

        y = data['Time_from_Pickup_to_Arrival']
        X = data.drop('Time_from_Pickup_to_Arrival', axis=1)
        return y,X
    
    y_base,X_base = yx_split(df)
    
    def train_validate_split(X,y):
        ''' Docstring'''

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)
        return X_train, y_train, X_test, y_test
    
    X_t_base, y_t_base, X_v_base, y_v_base = train_validate_split(X_base,y_base)
    
    def scale(X_t, X_v, method='standard'):
        ''' Docstring'''

        X_train = X_t.copy()
        X_val = X_v.copy()

        # Do not standardise dummy variable
        col_names = [col for col in X_train if col != 'Business_or_Personal']

        if method not in ['standard','normal']:
            raise ValueError('Method specified is invalid. Choose between normal or standard.')
        elif method == 'standard':
            s = StandardScaler()
            # Use standardised train X to scale test X, i.e fit on the train set and just transform the validation set
            X_train[col_names] = s.fit_transform(X_train[col_names])
            X_val[col_names] = s.transform(X_val[col_names])
        else:
            n = MinMaxScaler()
            X_train[col_names] = n.fit_transform(X_train[col_names])
            X_val[col_names] = n.transform(X_val[col_names])

        return X_train, X_val
    
    X_t_base_s, X_v_base_s = scale(X_t_base, X_v_base, method='standard')
    
    X_nouse, X_test = scale(X_t_base,df_t, method='standard')

    
    

    
    
    
    
    predict_vector = X_test
    
    #feature_vector_df[['Pickup Lat','Pickup Long',
                                        #'Destination Lat','Destination Long']]

    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
