import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


#Creating the Dataframe
df=pd.read_csv('/Users/uttarankanungo/Desktop/mobile_data_model/data/used_phone_price_prediction_1M copy 2.csv')

#Creating 'original_price_cat' column
df['original_price_cat']=pd.cut(df['original_price'],bins=5,labels=[1,2,3,4,5])

#Performing StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(df,df['original_price_cat']):
    train_set=df.loc[train_index].drop('original_price_cat',axis=1)
    test_set=df.loc[test_index].drop('original_price_cat',axis=1)

mobile_data_features=train_set.drop('resale_price',axis=1)
mobile_data_labels=train_set['resale_price']

cat_atr=['model','brand','os_type','condition','city_tier','seller_type']
num_atr=mobile_data_features.drop(['model','brand','os_type','condition','city_tier','seller_type'],axis=1).columns.to_list()

#Pipelines
num_pipeline=Pipeline([('scaler',StandardScaler())])
cat_pipeline=Pipeline([('ohe',OneHotEncoder(handle_unknown='ignore'))])
full_pipeline=ColumnTransformer([('num',num_pipeline,num_atr),
                                ('cat',cat_pipeline,cat_atr)])

final_mobile_data_features=full_pipeline.fit_transform(mobile_data_features)




#Selecting models based on cross validation score 
model_rf=RandomForestRegressor()
rf_rmses=-cross_val_score(model_rf,final_mobile_data_features,mobile_data_labels,scoring='neg_root_mean_squared_error',cv=5)
print(pd.Series(rf_rmses).describe())



model_lr=LinearRegression()
lr_rmses=-cross_val_score(model_lr,final_mobile_data_features,mobile_data_labels,scoring='neg_root_mean_squared_error',cv=10)
print(pd.Series(lr_rmses).describe())



model_dt=DecisionTreeRegressor()
dt_rmses=-cross_val_score(model_dt,final_mobile_data_features,mobile_data_labels,scoring='neg_root_mean_squared_error',cv=5)
print(pd.Series(dt_rmses).describe())

