import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def day_filter(frame, day_num):
    def day(row):
        if row.timestamp.day == day_num:
            return 1
        else:
            return 0

    frame['flag'] = frame.apply(day, axis=1)
    frame = frame[frame.flag != 0]
    frame = frame.drop(['flag'], axis=1)
    return frame


def thresholding(daylim, nightlim, frame, keepzeros=True):
    # 0 for drop, 1 for keep
    def limits(row, daylimit, nightlimit, zeros=True):
        if zeros != True and row.avg_speed == 0:
            return 0
        elif 18 >= row.timestamp.hour > 6 and row.avg_speed < daylimit:
            return 1
        elif (row.timestamp.hour > 18 or row.timestamp.hour <= 6) and row.avg_speed < nightlimit:
            return 1
        return 0

    frame["flag"] = frame.apply(limits, axis=1, args=[daylim, nightlim, keepzeros])
    frame = frame[frame.flag != 0]
    frame = frame.drop(['flag'], axis=1)
    return frame


def preprocess_cam_data(frame):
    frame['timestamp'] = pd.to_datetime(frame['timestamp'], unit='s')
    frame = day_filter(frame, 9)
    frame = frame[(frame['measurement_configuration.metric_type'] == 'AVG_SPEED')]
    frame['avg_speed'] = frame['value']
    frame = thresholding(daylim=70, nightlim=90, frame=frame, keepzeros=True)
    return frame


def preprocess_bus_data(frame):
    frame.rename(columns={'time_stamp': 'timestamp'}, inplace=True)
    frame['velocity'] = frame['velocity'].astype('int')
    frame['timestamp'] = pd.to_datetime(frame['timestamp'], unit='ms')
    frame = frame.sort_values(by='timestamp')
    return frame


def resample(frame, column_name, diff):
    frame = frame.set_index('timestamp')
    frame = frame.loc[~frame.index.duplicated(), :]
    frame.flags.allows_duplicate_labels = False
    str_ = str(diff) + 'S'

    frame['count'] = 1

    frame = frame.resample(str_).agg({
        column_name: np.mean,
    })
    frame = frame.reset_index()
    frame.flags.allows_duplicate_labels = True
    return frame


def two_line_plot(bus_frame, cam_frame):
    font = {'family': 'serif', 'size': 20}
    plt.plot(cam_frame['timestamp'], cam_frame['avg_speed'], label='cam')
    plt.plot(bus_frame['timestamp'], bus_frame['velocity'], label='bus')
    plt.xlabel("time", fontdict=font)
    plt.ylabel("km/h", fontdict=font)
    plt.ylim(0,55)
    plt.legend()
    plt.show()


def get_cam_column(column_name, frame, metric):
    frame = frame[(frame['measurement_configuration.metric_type'] == metric)]
    frame[column_name] = frame['value']
    frame = frame.drop(
        ['value', 'measurement_configuration.entity.entity_id', 'measurement_configuration.entity.entity_type',
         'measurement_configuration.metric_type'], axis=1)
    return frame


def create_joined_cam_data(frame, day):
    cam_speed = get_cam_column('avg_speed', day_filter(frame, day), 'AVG_SPEED')
    cam_flow = get_cam_column('total_flow', day_filter(frame, day), 'TOTAL_FLOW')
    cam_speed_resampled = resample(cam_speed, 'avg_speed', 15)
    cam_flow_resampled = resample(cam_flow, 'total_flow', 30)
    res = cam_speed_resampled.set_index('timestamp').join(cam_flow_resampled.set_index('timestamp'))
    res = res.reset_index()
    res = res.sort_values(by='timestamp')
    return res
