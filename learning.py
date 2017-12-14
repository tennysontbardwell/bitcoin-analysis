#!/usr/bin/env python3
from sklearn import linear_model, model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import luigi
import json
from os import path
import os
from functools import lru_cache
from settings import DATA_DIR, RES_DIR, HOURS_24, BITSTAMP_PRICE_FILE
from processing import FeaturizeTransactionInterval
import pickle
from pytz import timezone
import calendar
from datetime import datetime
from bokeh.plotting import figure, output_file, save


class Prices:
    def __init__(self, file_path):
        with open(path.join(DATA_DIR, BITSTAMP_PRICE_FILE)) as fp:
            self.data = pd.read_csv(fp)

    @lru_cache(maxsize=100000)
    def _convert_to_epoch(self, date_str):
        '''Bitstamp's data is in form YYYY-MM-DD, but each happened on EST at 6pm

        This function converts Bitstamp's date format to the actual time
        '''
        year,month,day = int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:10])
        est = timezone('US/Eastern').localize(datetime(year,month,day,18,0,0))
        return calendar.timegm((est.year, est.month, est.day,
                            est.hour, est.minute, est.second))

    def get_series(self, epoch_time):
        '''returns the nearest data point before or equal to epoch_time'''
        for _,series in self.data.iterrows():
            if self._convert_to_epoch(series['Date']) <= epoch_time:
                # print('Request {}, returned {}'.format(
                #     epoch_time,self._convert_to_epoch(series['Date'])))
                return series


class LearnPredictPrice(luigi.Task):
    start_time = luigi.IntParameter()
    end_time = luigi.IntParameter()
    interval_size = luigi.IntParameter()
    validation_splits = luigi.IntParameter(default=3)
    # in set {lasso, elasticnet}
    ml_technique = luigi.Parameter(default='lasso')  # TODO implement
    # in set {future, current}
    problem = luigi.Parameter(default='future')  # TODO implement

    def get_interval_vec(self, interval):
        with interval.output().open() as fp:
            features = json.load(fp)
            price_info_res = self.prices.get_series(
                interval.end_time + self.interval_size)
            price_info_history = \
                [self.prices.get_series(interval.end_time - days * self.interval_size)
                 for days in range(7)]
        if self.problem == 'future':
            return (
                [
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
                    # price_info_history[0]['High'],
                    # price_info_history[0]['Low'],
                    price_info_history[0]['Close'],
                    price_info_history[1]['Close'],
                    price_info_history[2]['Close'],
                    price_info_history[3]['Close'],
                    price_info_history[4]['Close'],
                    price_info_history[5]['Close']
                ],
                (price_info_res['Close'] - price_info_history[0]['Close']) / price_info_history[0]['Close'],
                {
                    'price': price_info_history[0]['Close'],
                    'title': 'Predicted Next Day Prices Against Actual, Using {}, Time Slice {}'
                }
            )

        elif self.problem == 'volume':
            return (
                [
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
                    price_info_res['High'],
                    price_info_res['Low'],
                    price_info_res['Close'],
                    price_info_history[0]['High'],
                    price_info_history[0]['Low'],
                    price_info_history[0]['Close'],
                    price_info_history[1]['Close'],
                    price_info_history[2]['Close'],
                    price_info_history[3]['Close'],
                    price_info_history[4]['Close'],
                    price_info_history[5]['Close']
                ],
                features['amount_traded'],
                {
                    'price': price_info_history[0]['Close'],
                    'title': 'Predicted Same Day Volume Against Actual, Using {}, Time Slice {}'
                }
            )

        elif self.problem == 'current':
            return (
                [
                    features['total_transactions'],
                    # features['total_inputs'],
                    # features['total_outputs'],
                    features['amount_traded'],
                    # features['old_amounts_traded']['0.1'] \
                    #         / features['amount_traded'],
                    features['old_amounts_traded']['0.2'] \
                            / features['amount_traded'],
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
                    features['old_amounts_traded']['0.8'] \
                            / features['amount_traded'],
                    # features['old_amounts_traded']['0.9'] \
                    #         / features['amount_traded'],
                ],
                (price_info_history[0]['Close'] - price_info_history[0]['Open']) / price_info_history[0]['Open'],
                {
                    'price': price_info_history[0]['Close'],
                    'title': 'Predicted Same Day Prices Against Actual, Using {}, Time Slice {}'
                }
            )

    def output(self):
        dir_name = 'learn_predict_price_from_{}_to_{}_intervals_{}_{}splits_{}_{}'.format(
                self.start_time, self.end_time, self.interval_size,
                self.validation_splits, self.ml_technique, self.problem)
        return {
            'training_data': luigi.LocalTarget(
                    path.join(RES_DIR,dir_name,'training_data.json')),
            'model': luigi.LocalTarget(
                    path.join(RES_DIR,dir_name,'model.pickle')),
            'results': luigi.LocalTarget(
                    path.join(RES_DIR,dir_name,'results.txt')),
            'plots': [luigi.LocalTarget(path.join(RES_DIR,dir_name,
                        'time_splice_{}.html'.format(x+1)))
                      for x in range(self.validation_splits)]
        }

    def requires(self):
        for i in range(self.start_time, self.end_time, self.interval_size):
            start = i
            end = i + self.interval_size
            if end > self.end_time:
                continue
            yield FeaturizeTransactionInterval(start_time=start, end_time=end)

    def run(self):
        model_filename = self.output()['model'].path
        if not path.exists(path.dirname(model_filename)):
            os.makedirs(path.dirname(model_filename))
        with self.output()['results'].open('w') as fp:
            fp.write('tmp data')

        self.prices = Prices(path.join(DATA_DIR, BITSTAMP_PRICE_FILE))
        X = []
        Y = []
        for interval in self.requires():
            x,y,feature_misc = self.get_interval_vec(interval)
            X.append(x)
            Y.append(y)

        with self.output()['training_data'].open('w') as fp:
            json.dump({'X': X, 'Y': Y}, fp, indent=4)
        X = np.array(X)
        Y = np.array(Y)

        with self.output()['results'].open('w') as fp:
            tscv = model_selection.TimeSeriesSplit(n_splits=self.validation_splits)
            for i,(train_index, test_index) in enumerate(tscv.split(X)):

                clf = {
                    'lasso': linear_model.Lasso(),
                    'elasticnet': linear_model.ElasticNet(),
                    'ridge': linear_model.Ridge()
                    # 'lasso': linear_model.LassoLarsCV(),
                    # 'elasticnet': linear_model.ElasticNetCV(
                    #     l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                    #     max_iter=10**6
                    # ),
                    # 'ridge': linear_model.RidgeCV(
                    #     alphas=[10**x for x in range(-2,3)]),
                }[self.ml_technique]

                ml_pretty_name = {
                    'lasso': 'Lasso',
                    'elasticnet': 'Elastic Net',
                    'ridge': 'Ridge Regression',
                }[self.ml_technique]

                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                clf.fit(x_train, y_train)
                test_res = clf.predict(x_test)

                fp.write('====== Score {} ======\n'.format(i+1))
                fp.write('{}\n'.format(clf.score(x_test, y_test)))

                fp.write('====== Importance {} ======\n'.format(i+1))
                try:
                    if 'coef_' in dir(clf):
                            fp.write(str(clf.coef_))
                    else:
                        fp.write(str(clf.feature_importances_) + '\n')
                except AttributeError:
                    pass

                fp.write('====== Test {} ======\n'.format(i+1))
                for a,b in zip(test_res, y_test):
                    fp.write('Predicted: {:8.2f}   Actual: {:8.2f}\n'.format(a,b))
                output_file(self.output()['plots'][i].path,
                            mode="cdn")
                TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"
                p = figure(
                    title=feature_misc['title'].format(ml_pretty_name, i+1),
                    x_axis_label='Actual',
                    y_axis_label='Predicted',
                    tools=TOOLS)
                p.circle(y_test, test_res, line_color=None)
                save(p)

            # names = ['total tx', 'total in', 'total out', 'amount traded',
            #          '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8'. '.9',
            #          'high', 'low', 'open', 'close']
            # rf = RandomForestRegressor()
            # rf.fit(x, y)
            # fp.write("Features sorted by their score:\n")
            # fp.write(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
            #             reverse=True))

        with open(model_filename, 'wb') as fp:
            pickle.dump(clf, fp)
