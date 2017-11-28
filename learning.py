#!/usr/bin/env python3
from sklearn import linear_model
import json
import luigi
from os import path
import os
from settings import RES_DIR, HOURS_24
from processing import FeaturizeTransactionInterval
import pickle


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
        vectors = []
        labels = []
        for interval in self.requires():
            with interval.output().open() as fp:
                features = json.load(fp)
                vectors.append([
                    features['total_transactions'],
                    features['total_inputs'],
                    features['total_outputs'],
                    features['old_amounts_traded']['0.1'] \
                            / features['amount_traded'],
                    features['old_amounts_traded']['0.2'] \
                            / features['amount_traded'],
                    features['old_amounts_traded']['0.3'] \
                            / features['amount_traded'],
                    features['old_amounts_traded']['0.4'] \
                            / features['amount_traded'],
                    features['old_amounts_traded']['0.5'] \
                            / features['amount_traded'],
                    features['old_amounts_traded']['0.6'] \
                            / features['amount_traded'],
                    features['old_amounts_traded']['0.7'] \
                            / features['amount_traded'],
                    features['old_amounts_traded']['0.8'] \
                            / features['amount_traded'],
                    features['old_amounts_traded']['0.9'] \
                            / features['amount_traded']
                ])
                labels.append(features['amount_traded'])

        clf = linear_model.Lasso(alpha=0.1)
        clf.fit(vectors, labels)

        filename = self.output()['model'].path
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, 'wb') as fp:
            pickle.dump(clf, fp)
        with self.output()['results'].open('w') as fp:
            fp.write('tmp data')

