import config

import pandas as pd
import numpy as np

class Data:

    def __init__(self):
        self.df_store = pd.read_csv('./store-sales/store.csv')
        self.df_train = pd.read_csv('./store-sales/train.csv', low_memory=False)

    @staticmethod
    def str_to_id(letter):
        if letter == '0':
            return 0
        elif letter == 'a':
            return 1
        elif letter == 'b':
            return 2
        elif letter == 'c':
            return 3
        elif letter == 'd':
            return 4
        else:
            return -1

    @staticmethod
    def AVGSales_to_categorical(avg_sales):
        """
        mean      5763.320541
        std       2046.447377
        min       2244.503185
        25%       4412.415567
        50%       5459.185775
        75%       6633.871550
        max      20718.515924
        """
        if avg_sales < 4413:
            return 1
        elif avg_sales < 5460:
            return 2
        elif avg_sales < 6634:
            return 3
        elif avg_sales < 20825:
            return 4
        else:
            return 0

    def preprocess_dataset(self):
        cut = int(config.PERCENTAGE_CUT_TRAIN_TEST * self.df_train.shape[0])
        self.df_store['AVGSales'] = np.array(self.df_train[:cut].groupby('Store')['Sales'].mean())
        self.df_store['AVGSalesCategorical'] = self.df_store['AVGSales'].apply(lambda avg_sales : self.AVGSales_to_categorical(avg_sales))

        self.df_preprocessed = self.df_train.merge(self.df_store, how='left', on='Store')

        self.df_preprocessed['Date'] = pd.to_datetime(self.df_preprocessed['Date'])
        self.df_preprocessed['Year'] = self.df_preprocessed.Date.dt.year
        self.df_preprocessed['Month'] = self.df_preprocessed.Date.dt.month
        self.df_preprocessed['Day'] = self.df_preprocessed.Date.dt.day
        self.df_preprocessed['WeekNB'] = self.df_preprocessed.Date.dt.isocalendar().week

        self.df_preprocessed = self.df_preprocessed[self.df_preprocessed.Open == 1].copy()

        self.df_preprocessed['CompetitionDuration'] = 12 * (self.df_preprocessed.Year - self.df_preprocessed.CompetitionOpenSinceYear) + (self.df_preprocessed.Month - self.df_preprocessed.CompetitionOpenSinceMonth)
        self.df_preprocessed['CompetitionDuration'] = self.df_preprocessed['CompetitionDuration'].apply(lambda x: 0 if x < 0 else x).fillna(0)

        self.df_preprocessed['Promo2Duration'] = 12 * (self.df_preprocessed.Year - self.df_preprocessed.Promo2SinceYear) + (self.df_preprocessed.WeekNB - self.df_preprocessed.Promo2SinceWeek) / 4
        self.df_preprocessed['Promo2Duration'] = self.df_preprocessed['Promo2Duration'].map(lambda x: 0 if x < 0 else x).fillna(0) * self.df_preprocessed['Promo2']

        self.df_preprocessed = self.df_preprocessed.sort_values(by=['Date', 'Store']).reset_index()

        max_distance = self.df_preprocessed.CompetitionDistance.max()
        self.df_preprocessed['CompetitionDistance'].fillna(max_distance * 2, inplace=True)

        self.df_preprocessed['StateHoliday'] = self.df_preprocessed['StateHoliday'].apply(lambda x: self.str_to_id(x))
        self.df_preprocessed['StoreType'] = self.df_preprocessed['StoreType'].apply(lambda x: self.str_to_id(x))
        self.df_preprocessed['Assortment'] = self.df_preprocessed['Assortment'].apply(lambda x: self.str_to_id(x))

        self.df_preprocessed.to_csv('preprocessed_dataset.csv')

    def get_preprocessed_df(self):
        print('Preprocessing data ...')
        self.preprocess_dataset()
        return self.df_preprocessed

    def get_store_df(self):
        return self.df_store
