"""
This script is created to compile county-level files of population and devices reporting.

Created 2/4/2021 by Ryan Harp.
"""


## Importing Modules
import numpy as np
import pandas as pd
import os



## Compiling a county-level file similar to home_panel_summary
df_home_panel_cbg = pd.read_csv('./test_data/home_panel_summary/2018-01.csv')
county_data = pd.DataFrame({'county': [], 'number_devices_residing': []})
len_ind = np.shape(df_home_panel_cbg)[0]
counties = np.ndarray(len_ind, dtype='object'); counties[:] = np.nan
cbg_devices_residing = np.ndarray(len_ind); cbg_devices_residing[:] = np.nan
for index, cbg in df_home_panel_cbg.iterrows():
    cbg_num = str(cbg['census_block_group'])
    if len(cbg_num) < 12:
        cbg_county = '0' + cbg_num[0:4]
    elif len(cbg_num) == 12:
        cbg_county = cbg_num[0:5]
    counties[index] = cbg_county
    cbg_devices_residing[index] = cbg['number_devices_residing']
county_data['county'] = pd.Series(counties)
county_data['number_devices_residing'] = pd.Series(cbg_devices_residing)
county_devices_residing = county_data.groupby(['county'], as_index=False).sum()

# adding in county_level total populations
df = pd.read_csv('./test_data/census_data/data/cbg_b01.csv')
df_cbg_pop = df.filter(['census_block_group', 'B01001e1'])  # total population for each cbg
county_data = pd.DataFrame({'county': [], 'county_population': []})
len_ind = np.shape(df_cbg_pop)[0]
counties = np.ndarray(len_ind, dtype='object'); counties[:] = np.nan
cbg_pops = np.ndarray(len_ind); cbg_pops[:] = np.nan
for index, cbg in df_cbg_pop.iterrows():
    cbg_num = str(cbg['census_block_group'])
    if len(cbg_num) < 12:
        cbg_county = '0' + cbg_num[0:4]
    elif len(cbg_num) == 12:
        cbg_county = cbg_num[0:5]
    counties[index] = cbg_county
    cbg_pops[index] = cbg['B01001e1']
county_data['county'] = pd.Series(counties)
county_data['county_population'] = pd.Series(cbg_pops)
county_pops = county_data.groupby(['county'], as_index=False).sum()
test = county_pops

# county_panel_summary = county_devices_residing.join(county_pops.set_index('county'), on='county')
# test.loc[(test['county'] == '55035')]  # testing with EC County

# looping over each .csv
files = []
for filename in os.listdir('/Users/ryanharp/Documents/Weather_and_Mobility/test_data/home_panel_summary'):
    files.append(filename)
files.sort()

for filename in files:
    if filename == '.DS_Store':
        continue
    df_home_panel_cbg = pd.read_csv('./test_data/home_panel_summary/' + filename)
    mon_str = filename[0:7]
    county_data = pd.DataFrame({'county': [], mon_str: []})
    len_ind = np.shape(df_home_panel_cbg)[0]
    counties = np.ndarray(len_ind, dtype='object')
    counties[:] = np.nan
    cbg_devices_residing = np.ndarray(len_ind)
    cbg_devices_residing[:] = np.nan
    for index, cbg in df_home_panel_cbg.iterrows():
        cbg_num = str(cbg['census_block_group'])
        if len(cbg_num) < 12:
            cbg_county = '0' + cbg_num[0:4]
        elif len(cbg_num) == 12:
            cbg_county = cbg_num[0:5]
        counties[index] = cbg_county
        cbg_devices_residing[index] = cbg['number_devices_residing']
    county_data['county'] = pd.Series(counties)
    county_data[mon_str] = pd.Series(cbg_devices_residing)
    county_devices_residing = county_data.groupby(['county'], as_index=False).sum()
    test = test.join(county_devices_residing.set_index('county'), on='county')

test.to_csv('./test_data/county_data/county_devices_residing.csv')
