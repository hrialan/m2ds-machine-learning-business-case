import data_access

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import math

data_access = data_access.Data()

df_preprocessed = data_access.get_preprocessed_df()

input_cols = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
                      'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionSince',
                                  'Day', 'Month', 'Year', 'WeekNB', 'Promo2', 'Promo2Since']

target_col = 'Sales'

inputs = df_preprocessed[input_cols].copy()
target = df_preprocessed[target_col].copy()

cut = int(0.7 * inputs.shape[0])

inputs_train, target_train = inputs[:cut], target[:cut]
inputs_test, target_test = inputs[cut:], target[cut:]

scaler = MinMaxScaler()
inputs_train_scaled = scaler.fit_transform(inputs_train)
inputs_test_scaled = scaler.fit_transform(inputs_test)

# LinearRegression
from sklearn.linear_model import LinearRegression
linear_regr = LinearRegression()
linear_regr.fit(inputs_train_scaled, target_train)

rmse = mean_squared_error(target_test, linear_regr.predict(inputs_test_scaled), squared=False)

print(' linear_regr Score : ' , linear_regr.score(inputs_test_scaled, target_test))
print(' linear_regr RMSE: ' , rmse)

# LogisticRegression
# from sklearn.linear_model import LogisticRegression
# logistic_regr = LogisticRegression()
# logistic_regr.fit(inputs_train_scaled, target_train)
#
# rmse = math.sqrt(mean_squared_error(target_test, logistic_regr.predict(inputs_test_scaled)))
#
# print(' logistic_regr Score : ' , logistic_regr.score(inputs_test_scaled, target_test))
# print(' logistic_regr RMSE: ' , rmse)


# RandomForestRegressor
# from sklearn.ensemble import RandomForestRegressor
# random_forest_regr = RandomForestRegressor()
# random_forest_regr.fit(inputs_train_scaled, target_train)
#
# rmse = math.sqrt(mean_squared_error(target_test, random_forest_regr.predict(inputs_test_scaled)))
#
# print(' random_forest_regr Score : ' , random_forest_regr.score(inputs_test_scaled, target_test))
# print(' random_forest_regr RMSE: ' , rmse)

# DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
decision_tree_regr = DecisionTreeRegressor()
decision_tree_regr.fit(inputs_train_scaled, target_train)

rmse = mean_squared_error(target_test, decision_tree_regr.predict(inputs_test_scaled), squared=False)

print(' decision_tree_regr Score : ' , decision_tree_regr.score(inputs_test_scaled, target_test))
print(' decision_tree_regr RMSE: ' , rmse)