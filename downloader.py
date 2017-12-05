import requests
from os import path
from time import sleep
import time
import sys
import logging
import json


LOG = logging.getLogger('downloader')
LOG.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
LOG.addHandler(ch)


DATA_PATH = 'data'
DATA_FILE = 'data.json'
# TIME_START = 1510000000  # Nov 6
TIME_START = 1512592000  # Dec 1
TIME_DELTA = 43200
DELAY = 0.1

getURL = lambda x: 'https://blockchain.info/rawblock/{}'.format(x)
getTimeURL = lambda t: 'https://blockchain.info/blocks/{}?format=json'.format(t)
LATEST_HASH_URL = 'https://blockchain.info/q/getblockcount'


class JsonFile:
    def __init__(self, file):
        self.file = file

    def __enter__(self):
        with open(self.file) as fp:
            self.j = json.load(fp)
        return self.j
    
    def __exit__(self, _, __, ___):
        self.store()

    def store(self):
        with open(self.file, 'w') as fp:
            json.dump(self.j, fp, indent=4)
        LOG.info('data file written')


def download(hash, height):
    file = path.join(DATA_PATH, str(height))
    if path.exists(file):
        LOG.info('file for {} already found'.format(height))
        return

    url = getURL(hash)
    r = requests.get(url)
    LOG.info('HASH #{}: {}, status code {}'.format(height, hash, r.status_code))
    assert r.status_code == 200
    with open(file, 'w') as fp:
        fp.write(r.text)

    sleep(DELAY)


def bulk_download(data_json, store):
    '''data_json has the following format:

    {
        'last_time_requested': some time w/o miliseconds,
        'hashes': {
            large_block_num_int: hash_val,
            large_block_num_int - 1: hash_val,
            large_block_num_int - 2: hash_val
        }
    }
    '''
    while True:
        t = data_json['last_time_requested'] - TIME_DELTA
        hashes = get_hashes(t)

        for hash in hashes['blocks']:
            download(hash['hash'], hash['height'])
            data_json[hash['height']] = hash['hash']

        data_json['last_time_requested'] = t
        store()


def get_latest():
    r = requests.get(LATEST_HASH_URL)
    return int(r.text)


def get_hashes(time):
    url = getTimeURL(time * 1000)
    r = requests.get(url)
    assert r.status_code == 200, str(r.status_code) + ' ' + r.text
    j = r.json()
    LOG.info('GOT {} hashes for time {}'.format(len(j['blocks']), time))
    return r.json()


def main():
    file = JsonFile(path.join(DATA_PATH, DATA_FILE)) 
    with file as data_json:
        if not data_json:
            data_json['last_time_requested'] = TIME_START
            data_json['hashes'] = {}
        bulk_download(data_json, file.store)


if __name__ == '__main__':
    main()

