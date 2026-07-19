import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

model_file = 'models/model.pkl'
pipeline_file = 'models/pipeline.pkl'

#Creating build_pipeline_function
def build_pipeline(num_atr, cat_atr):
    num_pipeline = Pipeline([('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))])
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_atr),
        ('cat', cat_pipeline, cat_atr)
    ])
    return full_pipeline

if not os.path.exists(model_file):
    df = pd.read_csv('/Users/uttarankanungo/Desktop/mobile_data_model/data/used_phone_price_prediction_1M copy 2.csv')
    df['original_price_cat'] = pd.cut(df['original_price'], bins=5, labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    #Performing Stratified Shuffle Split
    for train_index, test_index in split.split(df, df['original_price_cat']):
        train_set = df.loc[train_index].drop('original_price_cat', axis=1)
        test_set = df.loc[test_index].drop('original_price_cat', axis=1)

    #Training_Data
    mobile_data_features = train_set.drop('resale_price', axis=1)
    mobile_data_labels = train_set['resale_price']

    #Testing_Data
    test_data_features = test_set.drop('resale_price', axis=1)
    test_data_labels = test_set['resale_price']

    test_data_features.to_csv('data/input_mobile_resale_price.csv', index=False)

    cat_atr = ['model', 'brand', 'os_type', 'condition', 'city_tier', 'seller_type']
    num_atr = mobile_data_features.drop(cat_atr, axis=1).columns.to_list()

    #pipeline
    pipeline = build_pipeline(num_atr, cat_atr)
    final_mobile_data_features = pipeline.fit_transform(mobile_data_features)
    final_test_data_features = pipeline.transform(test_data_features)
    
    #Model Training
    model = RandomForestRegressor()
    model.fit(final_mobile_data_features, mobile_data_labels)
    
    #Prediction
    y_pred = model.predict(final_test_data_features)
    r2 = r2_score(test_data_labels, y_pred)
    print("R2 Score:", r2)

    joblib.dump(model, model_file)
    joblib.dump(pipeline, pipeline_file)

    print('model trained and saved')


#Interference Part
else:
    model = joblib.load(model_file)
    pipeline = joblib.load(pipeline_file)

    input_data = pd.read_csv('data/input_mobile_resale_price.csv')
    transformed_input = pipeline.transform(input_data)
    prediction = model.predict(transformed_input)
    input_data['resale_price'] = prediction
    input_data.to_csv('data/output_mobile_resale_price.csv', index=False)
    print('Interference completed,Result saved to output_mobile_resale_price.csv')
