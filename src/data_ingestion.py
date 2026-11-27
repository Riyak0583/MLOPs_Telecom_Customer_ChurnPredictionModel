import pandas as pd
def data_ingestion():
    '''
    * Title : "MLOps_Telecom_Customer_ChurnPredictionModel"

    * ML type : Classification 
    
    * Independent Column [X]: 'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges'

    * Dependent Columns [y] : 'Churn'
    
    * Problem Statements:
    1. To predict the churn rate
    2. Use Clustering approach to predict and classify the cluster pattern such as {l1,l2,l3}

   * accuracy : 79% 
    
    '''

    df = pd.read_csv(r'C:\MLOPs_Telecom_Customer_ChurnPredictionModel\data\raw\churn.csv')
    return df