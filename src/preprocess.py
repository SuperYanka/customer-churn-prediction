# src/preprocess.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Удаляем ID
    df = df.drop(columns=['customerID'])

    # Целевая переменная
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Преобразуем TotalCharges в число
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    # Числовые и категориальные признаки
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = df.drop(columns=numeric_features + ['Churn']).columns.tolist()

    # Разделение X и y
    X = df.drop(columns=['Churn']) #данные (без Churn)
    y = df['Churn']

    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Преобразование колонок
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor