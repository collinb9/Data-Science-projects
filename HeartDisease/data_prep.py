import pandas as pd
import numpy as np


def rename_columns(df):
    heading = {'age': 'age', 'sex': 'sex', 'cp':'chest pain type', 
                'trestbs': 'resting blood pressure', 'chol':'cholesterol', 
                'fbs':'fasting blood sugar > 120 mg/dl', 'restecg':'rest ecg results', 
                'thalach':'max heart rate', 'exang':'exercise induced angina', 
                'oldpeak':'ST depression (ecg)', 'slope':'slope of peak ST (ecg)', 
                'ca':'major vessels', 'thal': 'thalassemia', 'target':'target' }

    df.columns = [heading[i] for i in heading.keys()]
    return df
  
        
def separate_num_bin_cat_features(df):
    cat_features = ['chest pain type',
                    'rest ecg results',
                    'slope of peak ST (ecg)',
                    'thalassemia'
                    ]
    bin_features = ['sex', 'exercise induced angina', 'fasting blood sugar > 120 mg/dl']
    num_features = [feature for feature in df.drop(['target'], axis = 1).columns 
                    if feature not in cat_features if feature not in bin_features]
    return (num_features, bin_features, cat_features)
      
def rename_cat_values(df):
    df['chest pain type'].replace([0,1,2,3],['asymptomatic cp','atypical angina cp','non-anginal cp','typical angina cp'],inplace = True)
    df['rest ecg results'].replace([0,1,2],['left ventricular hypertrophy (ecg)', 'normal ecg', 'abnormal ecg'],inplace = True)
    df['slope of peak ST (ecg)'].replace([0,1,2],['downsloping','flat','upsloping'],inplace = True)
    df['thalassemia'].replace([1,2,3], ['fixed thal','no thal','reversable thal'], inplace = True)
 
  
 
 
 
 

if __name__ == '__main__':
    pass