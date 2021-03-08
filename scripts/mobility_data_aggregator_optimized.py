"""
This script will produce "true" visit totals by day for all POIs of a specified
NAICS code for one month. It is an optimized version (~50-100x faster) of the code implemented within
mobility_data_aggregator.py.

Created 3/8/2021 by Ryan Harp.
"""


#%% Initialization Steps

# importing modules
import numpy as np
import pandas as pd
import xarray as xr
import time

# initializing i/o variables
fname = '2018-01'
date_start = '2018-01-01'
date_end = '2018-01-31'


#%% Loading Data

# loading and filtering mobility data  # TODO: redo the preprocessing into HDF5 format
df = pd.read_csv('./test_data/mobility_data/2018-01/'+fname+'-part1.csv')  # TODO: be explicit about column types for loading
# df_mobility = pd.read_csv('/projects/b1045/Safegraph_v2/mobility_data/monthly_combined_files/'+fname+'_compiled.csv')
df_mobility = df.filter(['placekey', 'safegraph_place_id', 'naics_code', 'latitude', 'longitude', 'city', 'state',
        'region', 'date_range_start', 'date_range_end', 'raw_visit_counts', 'raw_visitor_counts', 'visits_by_day',
        'visitor_home_cbgs', 'poi_cbg'])
df_mobility = df.loc[(df_mobility['naics_code'] == 722511) | (df_mobility['naics_code'] == 512131)]  # filtering by relevant NAICS codes (722511 (restaurants) and 512131 (movie theaters) for now)
df_mobility = df_mobility.reset_index()
del df
# naics_list = [722511, 512131] # TODO: use in naics_list instead for filtering

# loading in cbg normalization data
df = pd.read_csv('./test_data/census_data/data/cbg_b01.csv')
df_cbg_pop = df.filter(['census_block_group', 'B01001e1'])  # total population for each cbg
del df

df = pd.read_csv('./test_data/home_panel_summary/'+fname+'.csv')
df_cbg_panel_size = df.filter(['census_block_group', 'number_devices_residing'])
del df

# loading in county normalization data
df = pd.read_csv('./test_data/county_data/county_devices_residing.csv')
# df['county'][df['county'].str.len() < 5] = '0' + df['county'][df['county'].str.len() < 5]
df_test = df.filter(['county', 'county_population', fname])
df_test = df_test['county_population']/df_test[fname]
df_county_pop_panel_size = dict(zip(df['county'], df_test))
del df
del df_test

# loading in cbg-month factors
df = pd.read_csv('./test_data/home_panel_summary/cbg_factors_by_month.csv', dtype={'cbg': str})
df['cbg'][df['cbg'].str.len() < 12] = '0' + df['cbg'][df['cbg'].str.len() < 12]
df_cbg_month_factors = dict(zip(df['cbg'].astype(str), df[fname]))
del df


#%% Analysis

t = time.time()

visits_by_day_str = df_mobility['visits_by_day'].astype('string')
visits_by_day_parse = visits_by_day_str.str.lstrip('[')
visits_by_day_parse = visits_by_day_parse.str.rstrip(']')
parsed = visits_by_day_parse.dropna()
test = pd.Series([np.array(x.split(',')).astype(int) for x in parsed], name='visits_by_day')
test2 = test.to_frame()
test2.index = parsed.index

visitors_by_month = df_mobility['raw_visitor_counts'].dropna().astype(int)
visits_by_month = df_mobility['raw_visit_counts'].dropna().astype(int)

poi_cbg = df_mobility['poi_cbg'].dropna().astype(int).astype(str)
poi_county = poi_cbg.str.slice(0, 5)
poi_county[poi_cbg.str.len() < 12] = '0' + poi_cbg.str.slice(0, 4)
poi_county = poi_county.astype(int)

poi_visitor_home_cbg = df_mobility['visitor_home_cbgs'].astype('string')
a = poi_visitor_home_cbg.dropna()
a = a.str.replace(':4,', ':3,')  # replacing counts of 4 with three since a value of 4 could represent 2, 3, or 4
a = a.str.replace(':4}', ':3}')
b = a.apply(lambda x: eval(x))
b = b.rename('dict')
c = b.apply(lambda x: list(x.keys()))
c = c.rename('visitor_home_cbgs')
df_combined = pd.concat([c, b], axis=1)

cbg_weight = b.apply(lambda x: sum(x.values()))
county_weight = visitors_by_month - cbg_weight
county_weight[county_weight < 0] = 0
poi_county_factor = poi_county.apply(lambda x: df_county_pop_panel_size.get(x))
# county_factor = poi_county.apply(lambda row: poi_county)

c_lim = pd.DataFrame(c)


test_d = c_lim.apply(lambda row: np.array(list(map(b[row.name].get, row[0]))), axis=1)
test_e = c_lim.apply(lambda row: np.array(list(map(df_cbg_month_factors.get, row[0]))), axis=1)
test_e_is_none = test_e.apply(lambda row: np.isnan(row.astype(float)).any())
# test_e[test_e_is_none] = test_e[test_e_is_none].apply(lambda row: row[row != None])
test_f = pd.DataFrame(test_e[test_e_is_none])
# test_d[test_e_is_none] = test_d[test_e_is_none].apply(lambda row: row[test_e[row.name]])
for index, row in test_f.iterrows():
    test_d[index] = test_d[index][test_e[index] != None]
    test_e[index] = test_e[index][test_e[index] != None]
test_g = test_d * test_e
test_h = (test_g.apply(np.sum, axis=0) + county_weight * poi_county_factor)/visits_by_month
test_h = test_h.rename('factors')

test_i = pd.concat([test2, test_h], axis=1)

#ctrue_visits_by_day = test2.apply(lambda row: test_h[row.name] * row[0])
true_visits_by_day = test_i['visits_by_day'] * test_i['factors']

elapsed = time.time() - t
print(elapsed)