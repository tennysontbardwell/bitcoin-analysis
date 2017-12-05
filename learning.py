#!/usr/bin/env python3
from sklearn import linear_model, model_selection
import pandas as pd
import numpy as np
import luigi
import json
from os import path
import os
from settings import DATA_DIR, RES_DIR, HOURS_24, BITSTAMP_PRICE_FILE
from processing import FeaturizeTransactionInterval
import pickle
from pytz import timezone
import calendar
from datetime import datetime


class Prices:
    def __init__(self, file_path):
        with open(path.join(DATA_DIR, BITSTAMP_PRICE_FILE)) as fp:
            self.data = pd.read_csv(fp)

    def _convert_to_epoch(self, date_str):
        '''Bitstamp's data is in form YYYY-MM-DD, but each happened on EST at 6pm

        This function converts Bitstamp's date format to the actual time
        '''
        year,month,day = int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10])
        est = timezone('US/Eastern').localize(datetime(year,month,day,18,0,0))
        return calendar.timegm((est.year, est.month, est.day,
                            est.hour, est.minute, est.second))

    def get_series(self, epoch_time):
        last = None
        for i,series in self.data.iterrows():
            if self._convert_to_epoch(series['Date']) < epoch_time:
                return series
            else:
                last = series


class LearnPredictPrice(luigi.Task):
    start_time = luigi.IntParameter()
    end_time = luigi.IntParameter()
    interval_size = luigi.IntParameter()

    def output(self):
        dir_name = 'learn_predict_price_from_{}_to_{}_intervals_{}'.format(
                self.start_time, self.end_time, self.interval_size)
        return {
            'model': luigi.LocalTarget(
                    path.join(RES_DIR,dir_name,'model.pickle')),
            'results': luigi.LocalTarget(
                    path.join(RES_DIR,dir_name,'results.txt'))
        }

    def requires(self):
        for i in range(self.start_time, self.end_time, self.interval_size):
            start = i
            end = i + self.interval_size
            if end > self.end_time:
                continue
            yield FeaturizeTransactionInterval(start_time=start, end_time=end)

    def run(self):
        prices = Prices(path.join(DATA_DIR, BITSTAMP_PRICE_FILE))
        x = []
        y = []
        for interval in self.requires():
            with interval.output().open() as fp:
                features = json.load(fp)
                price_info = prices.get_series(interval.start_time)
                price_info_res = prices.get_series(interval.end_time)
                x.append([
                    # features['total_transactions'],
                    # features['total_inputs'],
                    # features['total_outputs'],
                    # features['amount_traded'],
                    # features['old_amounts_traded']['0.1'] \
                    #         / features['amount_traded'],
                    # features['old_amounts_traded']['0.2'] \
                    #         / features['amount_traded'],
                    # features['old_amounts_traded']['0.3'] \
                    #         / features['amount_traded'],
                    # features['old_amounts_traded']['0.4'] \
                    #         / features['amount_traded'],
                    # features['old_amounts_traded']['0.5'] \
                    #         / features['amount_traded'],
                    # features['old_amounts_traded']['0.6'] \
                    #         / features['amount_traded'],
                    # features['old_amounts_traded']['0.7'] \
                    #         / features['amount_traded'],
                    # features['old_amounts_traded']['0.8'] \
                    #         / features['amount_traded'],
                    # features['old_amounts_traded']['0.9'] \
                    #         / features['amount_traded'],
                    price_info['High'],
                    price_info['Low'],
                    price_info['Open'],
                    price_info['Close']
                ])
                y.append(price_info['Open'] - price_info_res['Close'])
                # y.append(price_info_res['Close'])

        tscv = model_selection.TimeSeriesSplit(n_splits=3)
        x = np.array(x)
        y = np.array(y)

        for train_index, test_index in tscv.split(x):
            clf = linear_model.Lasso(alpha=0.1)
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf.fit(x_train, y_train)
            test_res = clf.predict(x_test)
            print('====== Test ======')
            for a,b in zip(test_res, y_test):
                print('Predicted: {:8.2f}   Actual: {:8.2f}'.format(a,b))
        import pdb; pdb.set_trace()

        filename = self.output()['model'].path
        if not path.exists(path.dirname(filename)):
            os.makedirs(path.dirname(filename))
        with open(filename, 'wb') as fp:
            pickle.dump(clf, fp)
        with self.output()['results'].open('w') as fp:
            fp.write('tmp data')

