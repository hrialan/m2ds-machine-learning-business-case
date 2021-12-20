import data_access
import config

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

try:
    regr = load('random_forest.joblib')
    print('✓ Model already trained \n')

except:
    print('Model not trained yet, Training in progress ... \n')
    regr = RandomForestRegressor()
    regr.fit(inputs_train_scaled, target_train)
    print('Saving model... \n')
    dump(regr, 'random_forest.joblib')

def predict_naive(input):
    df_store = data.get_store_df()
    pred = []
    for store_id in input['Store']:
        pred.append(df_store[df_store['Store'] == store_id]['AVGSales'].item())
    return pred


# print('R2 Score on test set  : ' , regr.score(inputs_test_scaled, target_test))
#
# pred_test = regr.predict(inputs_test_scaled)
#
# print('MSE Score on test set  : ' , mean_squared_error(target_test, pred_test))
#
# print('RMSE Score on test set  : ' , mean_squared_error(target_test, pred_test, squared=False))
#
# pred_test_naive = predict_naive(inputs_test)
#
# print('RMSE Score Naive model  : ' , mean_squared_error(target_test, pred_test_naive, squared=False))
#
# print('RMSE / AVG on test set % : ' ,(1 -  (mean_squared_error(target_test, pred_test, squared=False) / target_test.mean())) * 100 )
#
# print('RMSE / AVG Naive model % : ' ,(1 - (mean_squared_error(target_test, pred_test_naive, squared=False) / target_test.mean())) * 100)


import seaborn as sns
fig, ax = plt.subplots(figsize=(15, 5))
sns.set_theme(style="whitegrid")
data = {'input_cols': input_cols,
        'feature_importances': regr.feature_importances_}
df_fi = pd.DataFrame(data)
df_fi = df_fi.sort_values(by=['feature_importances'], ascending=False)
sns.barplot(x="feature_importances", y="input_cols", data=df_fi, ax=ax)

plt.show()

STORE_IDS = np.sort(inputs_test['Store'].unique())

compute_pred = False
if compute_pred:
    data_studio_ids = []
    data_studio_pred = []
    data_studio_date = []

    from datetime import timedelta, date

    start_date = date.today()

    for store_id in STORE_IDS:
        test_scaled = inputs_test_scaled[inputs_test['Store'] == store_id]
        date = start_date
        pred = regr.predict(test_scaled)
        for elt in pred:
            data_studio_ids.append(store_id)
            data_studio_date.append(date)
            data_studio_pred.append(elt)
            date = date + timedelta(days=1)

    df_data_studio = pd.DataFrame()
    df_data_studio['Store'] = data_studio_ids
    df_data_studio['Date'] = data_studio_date
    df_data_studio['Pred'] = data_studio_pred
    df_data_studio.to_csv('pred.csv', index=False)


compute_figures = False
if compute_figures:
    plt.figure()
    for store_id in STORE_IDS:
        print('Prediction for store ', store_id)
        test_scaled = inputs_test_scaled[inputs_test['Store'] == store_id]
        x = [i for i in range(len(test_scaled))]

        pred = regr.predict(test_scaled)
        true = target_test[inputs_test['Store'] == store_id]

        rmse = mean_squared_error(true, pred, squared=False)
        avg_sales_per_store = df_preprocessed[df_preprocessed['Store'] == store_id]['AVGSales'][0]

        plt.plot(x, pred, alpha = 0.7, label='Pred')
        plt.plot(x, true, alpha = 0.6, label='True')
        plt.title('Store ' + str(store_id) + ' / RMSE = ' + str(rmse))
        plt.xlabel('RMSE / AVG_STORE (%) = ' + str(round(rmse/avg_sales_per_store, 3) * 100))
        plt.legend()
        plt.savefig('figures/prediction_test_per_store/store_' + str(store_id) + '.png')
        plt.clf()
