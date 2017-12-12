import luigi
from settings import DATA_DIR, RES_DIR, TX_TIME_FILE, HOURS_24
from processing import FeaturizeTransactionInterval, MakeNetwork, get_tx_time,\
    get_blocks
from learning import LearnPredictPrice
from helpers import get_next_time


class Latest(luigi.WrapperTask):
    def requires(self):
        block, start, stop = get_blocks()
        # yield MakeNetwork(start_height=start, end_height=stop)
        start_time = get_next_time(get_tx_time(start), 18, 30, after=True)
        stop_time = get_next_time(get_tx_time(stop), 18, 30, after=False)
        # start_time, stop_time = 1505175181, 1510009531
        yield LearnPredictPrice(start_time=start_time, end_time=stop_time,
                interval_size=HOURS_24)
        # for time in range(start_time, stop_time - HOURS_24, HOURS_24):
        #     yield FeaturizeTransactionInterval(start_time=time,
        #                                        end_time=time+HOURS_24)
