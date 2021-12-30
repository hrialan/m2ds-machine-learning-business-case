import data_access
import config

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


data = data_access.Data()
df_preprocessed = data.get_preprocessed_df()

input_cols = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
                      'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionDuration',
                                  'Day', 'Month', 'Year', 'WeekNB', 'Promo2', 'Promo2Duration', 'AVGSalesCategorical']

target_col = 'Sales'

inputs = df_preprocessed[input_cols].copy()
target = df_preprocessed[target_col].copy()

cut = int(config.PERCENTAGE_CUT_TRAIN_TEST * inputs.shape[0])

inputs_train, target_train = inputs[:cut], target[:cut]
inputs_test, target_test = inputs[cut:], target[cut:]

scaler = MinMaxScaler()
inputs_train_scaled = scaler.fit_transform(inputs_train)
inputs_test_scaled = scaler.fit_transform(inputs_test)


# LinearRegression
from sklearn.linear_model import LinearRegression
linear_regr = LinearRegression()
linear_regr.fit(inputs_train_scaled, target_train)


# DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
decision_tree_regr = DecisionTreeRegressor()
decision_tree_regr.fit(inputs_train_scaled, target_train)

# pred_test = decision_tree_regr.predict(inputs_test_scaled)
#
# print('R2 Score on test set  : ' , decision_tree_regr.score(inputs_test_scaled, target_test))
#
# print('MSE Score on test set  : ' , mean_squared_error(target_test, pred_test))
#
# print('RMSE Score on test set  : ' , mean_squared_error(target_test, pred_test, squared=False))
#
# print('RMSE /AVG on test set : ' ,((mean_squared_error(target_test, pred_test, squared=False))/ target_test.mean()))
#
# print('RMSE / AVG on test set % : ' ,(1 -  (mean_squared_error(target_test, pred_test, squared=False) / target_test.mean())) * 100 )

# from sklearn.ensemble import RandomForestRegressor
# random_forest_regressor = RandomForestRegressor(max_depth=2)
# random_forest_regressor.fit(inputs_train_scaled, target_train)
#
# from sklearn import tree
# import matplotlib.pyplot as plt
# figure, axis = plt.subplots(1,figsize=(50,30))
# _ = tree.plot_tree(random_forest_regressor.estimators_[0], filled=True)
# plt.show()