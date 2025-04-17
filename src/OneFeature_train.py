import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split, learning_curve
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
# Import the plot_learning_curve function from the learning_curve.py file
from learning_curve import plot_learning_curve


def save_data(inputpath, outputpath='Dataset\\Bazate\\AnalizeSimplified'):
    df = pd.read_csv(inputpath)
    df = df.dropna(axis=1, how='all') 
    lst = ['donor_id', 'name', 'data', 'vaccine_response', 'visit_age', 'gender', 'race']
    for column in df.columns:
        if column not in lst:
            df.drop(column, axis=1, inplace=True) 
    df.to_csv(outputpath + '.csv', index=False)


def logistic_regression(inputpath, outputPath, plotPath, model_path):
    # Load data
    data = pd.read_csv(inputpath)  
    data = data[(data["vaccine_response"].isna() == False)]


    # Split data
    X = data.drop(columns=['vaccine_response', 'donor_id'])
    y = data['vaccine_response']


    # Define preprocessing for categorical and numerical columns
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Impute missing values and scale numeric features, OneHotEncode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
        ])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Create a pipeline with preprocessing and logistic regression
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # Fit the model
    model.fit(X_train, y_train)

    dump(model, model_path)

    # Plot learning curve
    plot_learning_curve(model, "Learning Curve (Logistic Regression)", X_train, y_train, cv=5, output_path=plotPath)

    # Predictions and evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

   


    # Save the classification report
    with open(outputPath, 'w', encoding='utf-8') as f:
        f.write('Train data procentage:'+ str(X_train.shape[0] / data.shape[0]) + "\n")
        f.write('Test data procentage:'+ str(X_test.shape[0] / data.shape[0]) + "\n")
        f.write("Dataset: " + inputpath +  "\n\n")

        f.write("Acuratețea modelului: " + str(accuracy) + "\n")
        f.write("\n\nRaport de clasificare:\n"+ report + "\n")


input_path_save = 'Dataset\\FluPRINT_database\\fluprint_export.csv'
save_data(input_path_save)

input_path1 = 'Dataset\\Bazate\\AnalizeSimplified.csv'
output_path1 = 'output\\OneFeature\\LR.txt'
plot_path1 = 'output\\OneFeature\\LR.png'
model_path1 = 'output\\OneFeature\\LR_1Feature.joblib'
logistic_regression(input_path1, output_path1, plot_path1, model_path1)
