#!/usr/bin/env python3
import json
import glob
from os import path
import networkx as nx
import luigi
import itertools
import pickle
from settings import DATA_DIR, RES_DIR, TX_TIME_FILE, HOURS_24


TX_TIME_CACHE = None


def get_blocks():
    blocks = [(int(path.basename(path_)), path_) for path_ in
                        glob.glob(DATA_DIR + '/[0-9]*')]
    blocks.sort()
    heights = [x for x,_ in blocks]
    a = min(heights)
    b = max(heights)
    i = a
    cut_at = 0
    for j,(height,size) in enumerate(blocks):
        if i != height:
            print('missing {} hash'.format(i))
            a = height
            i = height
            cut_at = j
        i += 1
    return blocks[cut_at:], a, b


def iter_blocks():
    blocks,_,_ = get_blocks()
    for size,path_ in blocks:
        with open(path_) as fp:
            yield (size, json.load(fp), len(blocks))


def get_txs(block_path):
    with open(block_path) as fp:
        j = json.load(fp)
    return j['tx']


def iter_transactions():
    for i,(size,dic,len_) in enumerate(iter_blocks()):
        for tx in dic['tx']:
            yield tx, {'current': i+1, 'total': len_}


def get_tx_time(block_height):
    global TX_TIME_CACHE
    file_loc = path.join(RES_DIR, TX_TIME_FILE)
    if not TX_TIME_CACHE:
        with open(file_loc) as fp:
            TX_TIME_CACHE = json.load(fp)

    if str(block_height) not in TX_TIME_CACHE:
        blocks,_,_ = get_blocks()
        added_blocks = set(TX_TIME_CACHE.keys())
        to_add_blocks = [(block,path_) for block,path_ in blocks
                         if str(block) not in added_blocks]
        print('Looking up times for {} blocks'.format(len(to_add_blocks)))
        for i,(block,path_) in enumerate(to_add_blocks):
            if (i+1) % 100 == 0:
                print('{}/{} done ({}%)'.format(i+1, len(to_add_blocks),
                                int((i+1) / len(to_add_blocks) * 100)))
            with open(path_) as fp:
                j = json.load(fp)
                TX_TIME_CACHE[block] = j['time']

        with open(file_loc, 'w') as fp:
            json.dump(TX_TIME_CACHE, fp)
        print('done')

    return TX_TIME_CACHE[str(block_height)]


class MakeNetwork(luigi.Task):
    start_height = luigi.IntParameter()
    end_height = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(path.join(RES_DIR,
            'network_from_{}_to_{}.pickle'.format(self.start_height, self.end_height)))

    def _iter_tx(self):
        total_num = self.end_height - self.start_height + 1
        for height in range(self.start_height, self.end_height + 1):
            with open(path.join(DATA_DIR, str(height))) as fp:
                for tx in json.load(fp)['tx']:
                    yield tx

            current_num = height - self.start_height + 1
            print('finished block {} ({}/{} blocks or {}%)'.format(
                    height, current_num, total_num,
                    int(current_num/total_num * 100)))

    def run(self):
        g = nx.Graph()
        for tx in self._iter_tx():

            inputs = []
            try:
                for input in tx['inputs']:
                    try:
                        inputs.append((
                            input['prev_out']['value'],
                            input['prev_out']['addr']))
                    except KeyError:
                        pass

                outs = []
                for out in tx['outs']:
                    try:
                        outs.append( (out['value'], out['addr']) )
                    except KeyError:
                        pass
                # total = sum( [out[0] for out in outs] )

                for input,out in itertools.product(inputs, outs):
                    val = input[0] * (out[0] / total)
                    g.add_edge(input[1], out[1], {'amount': val})
            except KeyError:
                pass

        with open(self.output().path, 'wb') as fp:
            pickle.dump(g, fp)


class FeaturizeTransactionInterval(luigi.Task):
    start_time = luigi.IntParameter()
    end_time = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(path.join(RES_DIR,
            'featurized_tx_interval_from_{}_to_{}.json'\
                    .format(self.start_time, self.end_time)))

    def run(self):
        before = False
        after = False
        txs = []
        for block,path_ in get_blocks()[0]:
            time = get_tx_time(block)
            if time <  self.start_time: before = True
            elif time >=  self.end_time: after = True
            else:
                txs += get_txs(path_)

        num_tx = len(txs)
        num_in = 0
        num_out = 0
        total_val = 0
        old_vals = {x:0 for x in
                    {'0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'}}

        for i,tx in enumerate(txs):
            num_in += len(tx.get('inputs', []))
            num_out += len(tx.get('out', []))
            for input_ in tx.get('inputs', []):
                val = 0
                try:
                    val = input_['prev_out']['value']
                except KeyError:
                    pass
                total_val += val
                try:
                    age = int(input_['prev_out']['tx_index']) / int(tx['tx_index'])
                    for k in old_vals.keys():
                        if age <= float(k):
                            old_vals[k] += val
                except KeyError:
                    pass

        features = {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'amount_traded': total_val,
            'total_transactions': num_tx,
            'total_inputs': num_in,
            'total_outputs': num_out,
            'old_amounts_traded': old_vals
        }
        with self.output().open('w') as fp:
            json.dump(features, fp)


def main():
    pass
if __name__ == '__main__':
    main()

