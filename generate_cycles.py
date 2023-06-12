import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch as th
import pickle as pkl

def generate_cycles(df):
    gaps = list((g:=df.timestamp.diff().gt(pd.Timedelta(minutes=1)))[g].index)
    cycles = []
    start = 0
    for gap in gaps:
        tdf = df.iloc[start:gap,:]
        comp_change = list((diffs:=tdf.COMP.diff().eq(-1))[diffs].index)
        cycles.extend([[comp_change[i],comp_change[i+1]] for i in range(len(comp_change)-1)])
        start = gap
    tdf = df.iloc[start:,:]
    comp_change = list((diffs:=tdf.COMP.diff().eq(-1))[diffs].index)
    cycles.extend([[comp_change[i],comp_change[i+1]] for i in range(len(comp_change)-1)])
    return cycles


def match_cycles_to_dates(cycle, df):
    cycle_start, cycle_end = cycle
    cycle_start_date = df.iloc[cycle_start,:].timestamp
    cycle_end_date = df.iloc[cycle_end,:].timestamp
    return [cycle_start_date, cycle_end_date]


def generate_cycle_tensors(df, cycle_inds, cols):
    scaler = StandardScaler()
    tensor_list = []
    for i_s, i_f in cycle_inds:
        df_slice = df.iloc[i_s:i_f,:]
        df_slice = df_slice.loc[:,cols]
        df_slice[df_slice.columns] = scaler.fit_transform(df_slice[df_slice.columns])
        tensor_chunk = th.tensor(df_slice.values).unsqueeze(0).float()
        tensor_list.append(tensor_chunk)
    return tensor_list

# This file assumes that there is a file called metropt2.csv in the same folder, downloaded from the zenodo link in the paper

metro = pd.read_csv("metropt2.csv")
correct_cols = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs',
                'Oil_temperature', 'Flowmeter', 'Motor_current','COMP', 'DV_eletric',
                'Towers', 'MPG', 'LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses']
orig_cols = ['oem_io.ANCH1', 'oem_io.ANCH2', 'oem_io.ANCH3',
             'oem_io.ANCH4', 'oem_io.ANCH5', 'oem_io.ANCH6', 'oem_io.ANCH7',
             'oem_io.ANCH8', 'oem_io.DI1', 'oem_io.DI2', 'oem_io.DI3', 'oem_io.DI4',
             'oem_io.DI5', 'oem_io.DI6', 'oem_io.DI7', 'oem_io.DI8']
metro.rename({orig_cols[i]: correct_cols[i] for i in range(len(correct_cols))}, inplace=True, axis=1)
metro["timestamp"] = pd.to_datetime(metro["timestamp"])

analog_sensors = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs',
                  'Oil_temperature', 'Flowmeter', 'Motor_current']

cycles = generate_cycles(metro)
all_cycles_dates = list(map(lambda x: match_cycles_to_dates(x, metro), cycles))
all_tensors_analog = generate_cycle_tensors(metro, cycles, analog_sensors)

train_start_date = metro.timestamp.iloc[0]
train_end_date = metro.timestamp.iloc[2658868]
test_start_date = metro.timestamp.iloc[2658869]
test_end_date = metro.timestamp.iloc[-1]

train_tensors = [all_tensors_analog[i] for i in range(len(all_tensors_analog)) if all_cycles_dates[i][0] < train_end_date]
test_tensors = [all_tensors_analog[i] for i in range(len(all_tensors_analog)) if all_cycles_dates[i][0] > test_start_date]


with open("data/final_train_tensors_analog_feats.pkl", "wb") as pklfile:
    pkl.dump(train_tensors, pklfile)

with open("data/final_test_tensors_analog_feats.pkl", "wb") as pklfile:
    pkl.dump(test_tensors, pklfile)
