#!/usr/bin/env python3
import json
import glob
from os import path
import networkx as nx
import luigi
import itertools
import pickle


DATA_DIR = 'data'
RES_DIR = 'tmp'


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


def iter_transactions():
    for i,(size,dic,len_) in enumerate(iter_blocks()):
        for tx in dic['tx']:
            yield tx, {'current': i+1, 'total': len_}


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


class Latest(luigi.WrapperTask):
    def requires(self):
        # block, start, stop = get_blocks()
        start, stop = 484743, 493386
        return MakeNetwork(start_height=start, end_height=stop)

def main():
    pass
if __name__ == '__main__':
    main()
