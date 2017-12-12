from pytz import timezone
import calendar
import time
from datetime import datetime, timedelta

def get_next_time(epoch, target_hour, target_minute, after=True):
    '''takes a time and target hour, returns next est with that hour

    if after = False then returns last time
    '''
    input_time = datetime.fromtimestamp(time.mktime(time.gmtime(epoch)))
    utc, est = timezone('UTC'), timezone('US/Eastern')
    input_est = est.normalize(utc.localize(input_time))
    target_time = datetime(input_est.year, input_est.month,
                           input_est.day, target_hour, target_minute, 0)
    target_time = est.localize(target_time)

    if (after and target_time >= input_est):
        pass
    elif (after and target_time < input_est):
        target_time += timedelta(days=1)
    elif (not after and target_time > input_est):
        target_time -= timedelta(days=1)
    elif (not after and target_time <= input_est):
        pass

    return calendar.timegm((target_time.year, target_time.month,
                           target_time.day, target_time.hour,
                           target_time.minute, target_time.second))
