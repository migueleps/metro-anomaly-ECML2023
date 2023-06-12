import numpy as np
import pickle as pkl
from operator import itemgetter
from itertools import groupby
import pandas as pd
from scipy.stats import zscore


def extreme_anomaly(dist):
    q25, q75 = np.quantile(dist, [0.25, 0.75])
    return q75 + 3*(q75-q25)

def simple_lowpass_filter(arr, alpha):
    y = arr[0]
    filtered_arr = [y]
    for elem in arr[1:]:
        y = y + alpha * (elem - y)
        filtered_arr.append(y)
    return filtered_arr


def detect_failures(anom_indices):
    failure_list = []
    failure = set()
    for i in range(len(anom_indices) - 1):
        if anom_indices[i] == 1 and anom_indices[i + 1] == 1:
            failure.add(i)
            failure.add(i + 1)
        elif len(failure) > 0:
            failure_list.append(failure)
            failure = set()

    if len(failure) > 0:
        failure_list.append(failure)

    return failure_list


def failure_list_to_interval(cycle_dates, failures):
    failure_intervals = []
    for failure in failures:
        failure = sorted(failure)
        failure_intervals.append(pd.Interval(cycle_dates[failure[0]][0], cycle_dates[failure[-1]][1], closed="both"))
    return failure_intervals


def collate_intervals(interval_list):
    diff_consecutive_intervals = [(interval_list[i+1].left - interval_list[i].right).days for i in range(len(interval_list)-1)]
    lt_1day = np.where(np.array(diff_consecutive_intervals) <= 1)[0]
    collated_intervals = []
    for k, g in groupby(enumerate(lt_1day), lambda ix: ix[0]-ix[1]):
        collated = list(map(itemgetter(1), g))
        collated_intervals.append(pd.Interval(interval_list[collated[0]].left, interval_list[collated[-1]+1].right, closed="both"))

    collated_intervals.extend([interval_list[i] for i in range(len(interval_list)) if i not in lt_1day and i-1 not in lt_1day])
    return sorted(collated_intervals)


def print_failures(cycle_dates, output):
    failures = detect_failures(output)
    failure_intervals = failure_list_to_interval(cycle_dates, failures)
    collated_intervals = collate_intervals(failure_intervals)
    for interval in collated_intervals:
        print(interval)


##### Results from the main paper #####

def generate_intervals(granularity, start_timestamp, end_timestamp):
    current_timestamp = start_timestamp
    interval_length = pd.offsets.DateOffset(**granularity)
    interval_list = []
    while current_timestamp < end_timestamp:
        interval_list.append(pd.Interval(current_timestamp, current_timestamp + interval_length, closed="left"))
        current_timestamp = current_timestamp + interval_length
    return interval_list


with open("data/training_chunk_dates.pkl", "rb") as chunk_dates_file:
    training_chunk_dates = pkl.load(chunk_dates_file)

with open("data/test_chunk_dates.pkl", "rb") as chunk_dates_file:
    test_chunk_dates = pkl.load(chunk_dates_file)


train_intervals = generate_intervals({"minutes": 5}, pd.Timestamp(training_chunk_dates[0][0]), pd.Timestamp(training_chunk_dates[-1][0]))
test_intervals = generate_intervals({"minutes": 5}, pd.Timestamp(test_chunk_dates[0][0]), pd.Timestamp(test_chunk_dates[-1][0]))


def map_cycles_to_intervals(interval_list, chunk_dates):
    cycles_dates = list(map(lambda x: pd.Interval(pd.Timestamp(x[0]), pd.Timestamp(x[1]), closed="both"), chunk_dates))
    return list(map(lambda x: np.where([x.overlaps(i) for i in cycles_dates])[0], interval_list))


train_chunks_to_intervals = map_cycles_to_intervals(train_intervals, training_chunk_dates)
test_chunks_to_intervals = map_cycles_to_intervals(test_intervals, test_chunk_dates)

alpha = 0.05

with open("results/final_chunks_complete_losses_WAE_LSTMDiscriminator_TCN_analog_feats_4_10_30_3_10.0_3_32_150_0.001_0.001_64.pkl", "rb") as loss_file:
    tl = pkl.load(loss_file)
    test_losses = tl["test"]
    train_losses = tl["train"]

median_train_losses = np.array([np.median(np.array(train_losses["reconstruction"])[tc]) for tc in train_chunks_to_intervals if len(tc) > 0])
median_test_losses = np.array([np.median(np.array(test_losses["reconstruction"])[tc]) for tc in test_chunks_to_intervals if len(tc) > 0])

median_train_critic = np.array([np.median(np.array(train_losses["critic"])[tc]) for tc in train_chunks_to_intervals if len(tc) > 0])
median_test_critic = np.array([np.median(np.array(test_losses["critic"])[tc]) for tc in test_chunks_to_intervals if len(tc) > 0])

combine_critic_reconstruction = np.abs(list(map(lambda x: np.nan_to_num(zscore(median_test_critic[:x], ddof=1)[-1]), 
                                                range(1,len(median_test_critic)+1)))) * median_test_losses
combine_critic_reconstruction_train = np.abs(list(map(lambda x: np.nan_to_num(zscore(median_train_critic[:x], ddof=1)[-1]), 
                                                      range(1,len(median_train_critic)+1)))) * median_train_losses

anom = extreme_anomaly(combine_critic_reconstruction_train)

binary_output = np.array(combine_critic_reconstruction > anom, dtype=int)

wae_gan_output = np.array(simple_lowpass_filter(binary_output,alpha))

print_failures(test_intervals, wae_gan_output)

with open("results/final_chunks_complete_losses_AE_tcn_ae_analog_feats_4_8_6_7_100_0.001_64.pkl", "rb") as loss_file:
    tl = pkl.load(loss_file)
    test_losses = tl["test"]
    train_losses = tl["train"]


median_train_losses = np.array(
    [np.median(np.array(train_losses)[tc]) for tc in train_chunks_to_intervals if len(tc) > 0])
median_test_losses = np.array([np.median(np.array(test_losses)[tc]) for tc in test_chunks_to_intervals if len(tc) > 0])

date_output_test = [interval.left for i, interval in enumerate(test_intervals) if len(test_chunks_to_intervals[i]) > 0]
date_output_train = [interval.left for i, interval in enumerate(train_intervals) if
                     len(train_chunks_to_intervals[i]) > 0]

anomaly_threshold = extreme_anomaly(median_train_losses)

binary_output = np.array(np.array(median_test_losses) > anomaly_threshold, dtype=int)

tcn_output = np.array(simple_lowpass_filter(binary_output, 0.03))

print_failures(test_intervals, tcn_output)

with open("results/final_chunks_complete_losses_AE_lstm_ae_analog_feats_4_5_150_0.001_64.pkl", "rb") as loss_file:
    tl = pkl.load(loss_file)
    test_losses = tl["test"]
    train_losses = tl["train"]

median_train_losses = np.array(
    [np.median(np.array(train_losses)[tc]) for tc in train_chunks_to_intervals if len(tc) > 0])
median_test_losses = np.array([np.median(np.array(test_losses)[tc]) for tc in test_chunks_to_intervals if len(tc) > 0])

date_output_test = [interval.left for i, interval in enumerate(test_intervals) if len(test_chunks_to_intervals[i]) > 0]
date_output_train = [interval.left for i, interval in enumerate(train_intervals) if
                     len(train_chunks_to_intervals[i]) > 0]

anomaly_threshold = extreme_anomaly(median_train_losses)

binary_output = np.array(np.array(median_test_losses) > anomaly_threshold, dtype=int)

lstm_output = np.array(simple_lowpass_filter(binary_output, 0.05))

print_failures(test_intervals, lstm_output)