from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from data_ingestion import data_ingestion

def data_preprocessing(df):
    '''
    Dataset: MLOPs_Telecom_Customer_ChurnPredictionModel
    
    Label_Column(X) : customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
                        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                        'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    
    Target_Column(y) : 'Churn'
    
    Steps :  
    1. Drop duplicate records.
    2. Feature Engineering (Optional).
    3. Separate independent (X) and dependent (y) columns.
    4. Split data into training and testing sets.
    5. Segregate numerical and categorical columns.
    6. Apply imputation and scaling to numerical columns.
    7. Apply imputation and encoding to categorical columns.
    8. Combine pipelines using ColumnTransformer.
    9. Fit and transform training data; transform testing data.
    10. Return preprocessed datasets.
    
    '''
# Step 1 : Drop duplicate columns
    df = df.drop_duplictes()
    print("Duplicates:" ,df.shape)

# Step 2 : Feature Engineering(Optional)

# Step 3 : Separate independent(y) and dependent(X) columns
     
    X = df.drop(columns=['Churn'])
    y = df['Churn']

# Step 4 : Split the dataset into train(seen) and test(unseen) 
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                     test_size=0.3,
                                                     random_state=1)
    
# Step 5 : Segreggate the pipeline into numerical and categorical columns
    
    numerical_co = X_train.select_dtypes(exclude = 'object').columns
    categorical_col = X_train.select_dtypes(include = 'object').columns

# Step 6 : Use numerical_pipeline

    numerical_co = Pipeline([('Imputation',SimpleImputer(strategy = 'median')),
                             ('Scaling',MinMaxScaler())])

# Step 7 : Use categorical_pipeline
    
    categorical_col = Pipeline([('Imputation',SimpleImputer(strategy = 'most_frequent')),
                                 ('Encoding',OneHotEncoder(handle_unknown='ignore'))])
    
    
# Step 8 : Combine pipeline using ColumnsTransformer to concat numerical and categorical columns

    tranformer = ColumnTransformer([('numerical_pipeline',
                                     'numerical_pipeline',
                                     'numerical_col'),
                                     ('categorical_pipeline',
                                      'categorical_pipeline',
                                      'categorical_col')])
    
# Step 9 : fit the transformer in seen and unseen data
    
    X_train = tranformer.fit_transform(X_train) # Seen data
    X_train = tranformer.tranform(X_test)       # Unseen data

# Step 10 : Return the preprocess data values
   
    X_train,X_test,y_train,y_test