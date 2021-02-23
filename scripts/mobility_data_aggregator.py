"""
This script will produce "true" visit totals by day for all POIs of a specified
NAICS code for one month.

Created 2/4/2021 by Ryan Harp.
"""


## importing modules
import numpy as np
import pandas as pd
import xarray as xr
import time

fname = '2018-01'
date_start = "2018-01-01"
date_end = "2018-01-31"

# limiting data for processing
df_mobility = pd.read_csv('/projects/b1045/Safegraph_v2/mobility_data/monthly_combined_files/'+fname+'_compiled.csv')
#df_mobility = pd.read_csv('/projects/b1045/Safegraph_v2/mobility_data/2018-01/'+fname+'-part10.csv')
df_filter = df_mobility.filter(['placekey', 'safegraph_place_id', 'naics_code', 'latitude', 'longitude', 'city', 'state',
        'region', 'date_range_start', 'date_range_end', 'raw_visit_counts', 'raw_visitor_counts', 'visits_by_day',
        'visitor_home_cbgs', 'poi_cbg'])
df_filter = df_filter.loc[(df_mobility['naics_code'] == 722511) | (df_mobility['naics_code'] == 512131)]  # filtering by relevant NAICS codes (722511 (restaurants) and 512131 (movie theaters) for now)
del df_mobility
# df_filter.to_csv('test_data/mobility_data/2018-01_filtered.csv', index = False)

# loading in cbg normalization data
df = pd.read_csv('/projects/b1045/Safegraph_v2/census_data/data/cbg_b01.csv')
df_pop = df.filter(['census_block_group', 'B01001e1'])  # total population for each cbg
del df

df = pd.read_csv('/projects/b1045/Safegraph_v2/home_panel_summary/'+fname+'.csv')
df_cbg_sample_size = df.filter(['census_block_group', 'number_devices_residing'])
del df

# loading in county normalization data
df = pd.read_csv('/projects/b1045/Safegraph_v2/county_data/county_devices_residing.csv')
df_county_sample_size = df.filter(['county', 'county_population', fname])
del df


def get_daily_true_visits(i):
    visitor_home_cbgs, raw_visit_count, raw_visitor_count, visits_by_day, poi_cbg, poi_county = get_poi_month_data(i)
    if type(visits_by_day) != np.ndarray or poi_county == '-99999':
        true_visits_by_day = np.nan
        return true_visits_by_day
    poi_county_pop, poi_county_devices_residing = get_poi_county_data(poi_county, df_county_sample_size)
    cbg_visits_adjusted, cbg_weight = calc_visitor_cbg(visitor_home_cbgs, df_cbg_sample_size, df_pop)
    remaining_visitors, remaining_adjusted = calc_visitor_poi_home_county(raw_visitor_count, cbg_weight, poi_county_devices_residing, poi_county_pop)
    cbg_visits_adjusted.append(remaining_adjusted)
    cbg_visits_factor = np.sum(cbg_visits_adjusted) / raw_visitor_count
    true_visits_by_day = visits_by_day * cbg_visits_factor
    return true_visits_by_day


def get_poi_month_data(i):
    visitor_home_cbgs = df_filter.iloc[i]["visitor_home_cbgs"]
    raw_visit_count = df_filter.iloc[i]["raw_visit_counts"]
    raw_visitor_count = df_filter.iloc[i]["raw_visitor_counts"]
    visits_by_day_str = df_filter.iloc[i]["visits_by_day"]
    if type(visits_by_day_str) != str:
        visits_by_day = np.nan
        poi_cbg = np.nan
        poi_county = np.nan
    else:
        visits_by_day = np.array(visits_by_day_str[1:-1].split(',')).astype(int)
        if np.isnan(df_filter.iloc[i]["poi_cbg"]):
            poi_cbg = '-999999999999'
            poi_county = '-99999'
        else:
            poi_cbg = str(int(df_filter.iloc[i]["poi_cbg"]))
        if len(poi_cbg) < 12:
            poi_county = '0' + poi_cbg[0:4]
        elif len(poi_cbg) == 12:
            poi_county = poi_cbg[0:5]
    return visitor_home_cbgs, raw_visit_count, raw_visitor_count, visits_by_day, poi_cbg, poi_county


def get_poi_county_data(poi_county, county_pops):
    poi_county_pop = county_pops.loc[(county_pops['county'] == float(poi_county))]['county_population']
    poi_county_devices_residing = county_pops.loc[(county_pops['county'] == float(poi_county))][
        fname]
    return poi_county_pop.iloc[0], poi_county_devices_residing.iloc[0]


def calc_visitor_cbg(visitor_home_cbgs, cbg_sample_size_df, cbg_pop_df):
    cbg_visits_adjusted = []
    cbg_weight = []
    if visitor_home_cbgs == '{}':
        return cbg_visits_adjusted, cbg_weight
    for visitor_home_cbg_str in visitor_home_cbgs[1:-1].split(','):
        visitor_home_cbg_num = np.int(visitor_home_cbg_str[1:13])
        raw_visit_home_cbg = np.int(visitor_home_cbg_str[-1])
        if raw_visit_home_cbg < 5:
            raw_visit_home_cbg = 3  # assuming 3 if raw_visits is less than 4
        cbg_weight.append(raw_visit_home_cbg)
        visits_adjusted = get_cbg_visits_adjusted(visitor_home_cbg_num, raw_visit_home_cbg, cbg_sample_size_df, cbg_pop_df)
        cbg_visits_adjusted.append(visits_adjusted)
        if visits_adjusted == -999:
            del cbg_weight[-1]
            del cbg_visits_adjusted[-1]
    return cbg_visits_adjusted, cbg_weight


def get_cbg_visits_adjusted(home_cbg_num, raw_visits, cbg_sample_size_df, cbg_pop_df):
    cbg_sample_size = cbg_sample_size_df.loc[(cbg_sample_size_df['census_block_group'] == home_cbg_num)]['number_devices_residing']
    cbg_pop = cbg_pop_df.loc[(cbg_pop_df['census_block_group'] == home_cbg_num)]['B01001e1']
    if cbg_pop.empty or cbg_sample_size.empty:
        return -999
    home_cbg_visits_adjusted = raw_visits / cbg_sample_size.iloc[0] * cbg_pop.iloc[0]
    return home_cbg_visits_adjusted


def calc_visitor_poi_home_county(raw_visitor_count, cbg_weight, poi_county_devices_residing, poi_county_pop):
    remaining_visitors = np.int(raw_visitor_count - sum(cbg_weight))
    remaining_adjusted = remaining_visitors / poi_county_devices_residing * poi_county_pop
    return remaining_visitors, remaining_adjusted


# prepping to run at scale
len_ind = np.shape(df_filter)[0]
poi_mon_data = pd.DataFrame({'placekey': [], 'naics_code': [], 'latitude': [], 'longitude': [], 'city': [], 'state': []})
placekey = np.ndarray(len_ind, dtype='object'); placekey[:] = np.nan
naics_code = np.ndarray(len_ind, dtype='object'); naics_code[:] = np.nan
latitude = np.ndarray(len_ind); latitude[:] = np.nan
longitude = np.ndarray(len_ind); longitude[:] = np.nan
city = np.ndarray(len_ind, dtype='object'); city[:] = np.nan
state = np.ndarray(len_ind, dtype='object'); state[:] = np.nan
true_visits_by_day = np.ndarray(shape=(len_ind, 31)); true_visits_by_day[:] = np.nan

t = time.time()
for index in np.arange(len_ind):
#for index in np.arange(26884, len_ind):
    if index % 1000 == 0:
        print(np.round(index/len_ind*100, 2))
        print(time.time() - t)
    placekey[index] = df_filter.iloc[index]['placekey']
    naics_code[index] = df_filter.iloc[index]['naics_code']
    latitude[index] = df_filter.iloc[index]['latitude']
    longitude[index] = df_filter.iloc[index]['longitude']
    city[index] = df_filter.iloc[index]['city']
    state[index] = df_filter.iloc[index]['region']
    if state[index] in ['VI', 'AS', 'GU']:
        true_visits_by_day[index] = np.nan
        continue
    true_visits_by_day[index] = np.round(get_daily_true_visits(index), 2)

poi_mon_data['placekey'] = pd.Series(placekey)
poi_mon_data['naics_code'] = pd.Series(naics_code)
poi_mon_data['latitude'] = pd.Series(latitude)
poi_mon_data['longitude'] = pd.Series(longitude)
poi_mon_data['city'] = pd.Series(city)
poi_mon_data['state'] = pd.Series(state)
# poi_mon_data['true_visits_by_day'] = pd.DataFrame(true_visits_by_day)
poi_mon_data.to_csv('/home/rdh0715/weather_and_mobility/'+fname+'.csv')

date = pd.date_range(start=date_start, end=date_end)
# true_visits_by_day_ds = xr.DataArray(true_visits_by_day, coords=[np.arange(len_ind), date], dims=["index", "date"])
true_visits_by_day_ds = xr.DataArray(true_visits_by_day, coords=[placekey, date], dims=["placekey", "date"])
true_visits_by_day_ds.to_netcdf('/home/rdh0715/weather_and_mobility/'+fname+'.nc')

elapsed = time.time() - t
print(elapsed)
print(index)
