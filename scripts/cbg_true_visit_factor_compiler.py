"""
This script compiles the factors needed to convert Safegraph visits to "true" visits for each
CBG-month.

inputs:
    1) number of devices residing in a CBG each month
    2) CBG population
outputs:
    1) file of factors for each CBG-month

Created 2/23/2021 by Ryan Harp.
"""

## importing modules
import pandas as pd
import numpy as np
import os
import time


## reading input files
# census block group sample size
df = pd.read_csv('./test_data/home_panel_summary/2018-01.csv')
df_cbg_sample_size = df.filter(['census_block_group', 'number_devices_residing'])
del df

# census block group population
df = pd.read_csv('./test_data/census_data/data/cbg_b01.csv')
df_cbg_pop = df.filter(['census_block_group', 'B01001e1'])  # total population for each cbg
del df


## function to compute cbg-month factors
def get_cbg_factor(home_cbg_num):
    cbg_sample_size = df_cbg_sample_size.loc[(df_cbg_sample_size['census_block_group'] == home_cbg_num)]['number_devices_residing']
    cbg_pop = df_cbg_pop.loc[(df_cbg_pop['census_block_group'] == home_cbg_num)]['B01001e1']
    if cbg_pop.empty or cbg_sample_size.empty:
        return -999
    return cbg_pop.iloc[0] / cbg_sample_size.iloc[0]


## main script
# looping over each .csv
files = []
for filename in os.listdir('/Users/ryanharp/Documents/Weather_and_Mobility/test_data/home_panel_summary'):
    files.append(filename)
files.sort()

first = True
len_ind = np.shape(df_cbg_pop)[0]
t = time.time()
for filename in files:
    print(filename)
    print(time.time() - t)
    if filename == '.DS_Store':
        continue
    df_cbg_sample_size = pd.read_csv('./test_data/home_panel_summary/' + filename)
    mon_str = filename[0:7]
    cbg_data = pd.DataFrame({'cbg': [], mon_str: []})
    cbgs = np.ndarray(len_ind, dtype='object')
    cbgs[:] = np.nan
    cbg_factor = np.ndarray(len_ind)
    cbg_factor[:] = np.nan
    for index, row in df_cbg_pop.iterrows():
        cbg_num = int(row['census_block_group'])
        cbgs[index] = cbg_num
        cbg_factor[index] = get_cbg_factor(cbg_num)
    cbg_data['cbg'] = pd.Series(cbgs)
    cbg_data[mon_str] = pd.Series(cbg_factor)
    if first:
        output = cbg_data
        first = False
    else:
        output = output.join(cbg_data.set_index('cbg'), on='cbg')
print(time.time() - t)

output.to_csv('./test_data/home_panel_summary/cbg_factors_by_month.csv')
