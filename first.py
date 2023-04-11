import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from joblib import dump, load

housing =pd.read_csv("data.csv")
##ploting the graph
# housing.hist(bins=50,figsize=(20,15))
# plt.show()
##long process to split data
# def split_train_test(data,test_ratio):
#     np.random.seed(42)
#     shuffled =np.random.permutation(len(data))
#     test_set_size= int(len(data)*test_ratio)
#     test_indices=shuffled[:test_set_size]
#     train_indices=shuffled[test_set_size:]
#     return data.iloc[train_indices],data.iloc[test_indices]

train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
# print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}")
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set= housing.loc[train_index]
    strat_test_set= housing.loc[test_index]
# print(strat_test_set.info)
## looking for correlation
corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

attributes=["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))
# plt.show()
housing= strat_train_set.drop("MEDV",axis=1)
housing_labels=strat_train_set["MEDV"].copy()
imputer= SimpleImputer(strategy="median")
imputer.fit(housing)
# print(imputer.statistics_)

x=imputer.transform(housing) 
housing_tr=pd.DataFrame(x,columns=housing.columns)
# print(housing_tr.describe())
my_pipeline= Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler()),
    ])
housing_num_tr= my_pipeline.fit_transform(housing)
# model= LinearRegression()
model= DecisionTreeRegressor()

model.fit(housing_num_tr,housing_labels)

some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
model.predict(prepared_data)
##evalutaing the model
housing_predicitions=model.predict(housing_num_tr)
mse= mean_squared_error(housing_labels,housing_predicitions)
rmsse=np.sqrt(mse)
scores= cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)



def print_scores(scores):
    print("scores:",scores)
    print("mean:",scores.mean())
    print("standard deviation :",scores.std())

print_scores(rmse_scores)

dump(model,'Dragon.joblib')

# testing

x_test=strat_test_set.drop("MEDV",axis=1)
y_test=strat_test_set["MEDV"].copy()
x_test_prepared=my_pipeline.transform(x_test)
final_predicitions=model.predict(x_test_prepared)
final_mse= mean_squared_error(y_test,final_predicitions)
final_rmse=np.sqrt(final_mse)
print(final_rmse)
print(final_predicitions,list(y_test))
