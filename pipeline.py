import luigi
from settings import DATA_DIR, RES_DIR, TX_TIME_FILE, HOURS_24
from processing import FeaturizeTransactionInterval, MakeNetwork, get_tx_time,\
    get_blocks
from learning import LearnPredictPrice
from helpers import get_next_time
import itertools


class Latest(luigi.WrapperTask):
    def requires(self):
        # Dec 1 EST
        d = (496967, 497148)
        # Dec 1, 12 am - 1 am EST
        d_hr = (496967, 496972)
        yield MakeNetwork(start_height=d_hr[0], end_height=d_hr[1])

        block, start, stop = get_blocks()
        start_time = get_next_time(get_tx_time(start), 18, 0, after=True)
        stop_time = get_next_time(get_tx_time(stop), 18, 0, after=False)
        # start_time, stop_time = 1505175181, 1510009531
        for ml, p in itertools.product(
                [
                    'lasso',
                    'elasticnet',
                    # 'svr-linear',
                    # 'svr-rbf',
                    'ridge'
                ],
                ['future', 'current', 'volume']):
            yield LearnPredictPrice(start_time=start_time, end_time=stop_time,
                                    interval_size=HOURS_24, ml_technique=ml,
                                    problem=p)
        # for time in range(start_time, stop_time - HOURS_24, HOURS_24):
        #     yield FeaturizeTransactionInterval(start_time=time,
        #                                        end_time=time+HOURS_24)
