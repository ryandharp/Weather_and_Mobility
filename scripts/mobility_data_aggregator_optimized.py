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

# parallelizing
import multiprocessing as mp

# putting a wrapper around the script
def wrapper(fname):

    # initializing i/o variables
    # fname = '2018-01'
    first_date_of_month = pd.Timestamp(fname)
    days_in_month = first_date_of_month.days_in_month
    date_range = pd.date_range(first_date_of_month, periods=days_in_month)

    naics_list = [512131, 445110, 622110, 531120, 722511, 452319, 722513, 721110, 722515, 722410, 488119]  # naics categories to examine


    #%% Loading Data

    # loading and filtering mobility data  # TODO: redo the preprocessing into HDF5 format
    # df = pd.read_csv('./test_data/mobility_data/2018-01/'+fname+'-part1.csv')  # TODO: be explicit about column types for loading
    df = pd.read_csv('/projects/b1045/Safegraph/mobility_data/monthly_combined_files/'+fname+'_compiled.csv')
    df_mobility = df.filter(['placekey', 'safegraph_place_id', 'naics_code', 'latitude', 'longitude', 'city', 'state',
            'region', 'date_range_start', 'date_range_end', 'raw_visit_counts', 'raw_visitor_counts', 'visits_by_day',
            'visitor_home_cbgs', 'poi_cbg'])
    df_mobility = df_mobility[df_mobility['naics_code'].isin(naics_list)]
    # df_mobility['poi_cbg'] = df_mobility['poi_cbg'].astype(str)
    # df_mobility.loc[df_mobility['poi_cbg'].str.len() < 12, 'poi_cbg'] = '0' + df_mobility['poi_cbg'][df_mobility['poi_cbg'].str.len() < 12]
    df_mobility = df_mobility.reset_index()
    del df

    # loading in cbg normalization data
    # df = pd.read_csv('./test_data/census_data/data/cbg_b01.csv')
    df = pd.read_csv('/projects/b1045/Safegraph/census_data/data/cbg_b01.csv')
    df_cbg_pop = df.filter(['census_block_group', 'B01001e1'])  # total population for each cbg
    del df

    # df = pd.read_csv('./test_data/home_panel_summary/'+fname+'.csv')
    df = pd.read_csv('/projects/b1045/Safegraph/home_panel_summary/'+fname+'.csv')
    df_cbg_panel_size = df.filter(['census_block_group', 'number_devices_residing'])
    del df

    # loading in county normalization data
    # df = pd.read_csv('./test_data/county_data/county_devices_residing.csv')
    df = pd.read_csv('/projects/b1045/Safegraph/county_data/county_devices_residing.csv')
    # df['county'][df['county'].str.len() < 5] = '0' + df['county'][df['county'].str.len() < 5]
    df_county_pop_panel_size = df.filter(['county', 'county_population', fname])
    df_county_factor = df_county_pop_panel_size['county_population'] / df_county_pop_panel_size[fname]
    county_factor_lookup = dict(zip(df_county_pop_panel_size['county'], df_county_factor))
    del df, df_county_pop_panel_size, df_county_factor

    # loading in cbg-month factors
    # df = pd.read_csv('./test_data/home_panel_summary/cbg_factors_by_month.csv', dtype={'cbg': str})
    df = pd.read_csv('/projects/b1045/Safegraph/home_panel_summary/cbg_factors_by_month.csv', dtype={'cbg': str})
    df_cbg_month_factors = df.copy()
    df_cbg_month_factors.loc[df['cbg'].str.len() < 12, 'cbg'] = '0' + df['cbg'][df['cbg'].str.len() < 12]
    cbg_month_factors_lookup = dict(zip(df_cbg_month_factors['cbg'].astype(str), df[fname]))
    del df, df_cbg_month_factors


    #%% Preprocessing

    # t = time.time()

    # pulling and parsing visits_by_day string
    visits_by_day_str = df_mobility['visits_by_day'].astype('string')
    visits_by_day_str_parsed = visits_by_day_str.str.lstrip('[')
    visits_by_day_str_parsed = visits_by_day_str_parsed.str.rstrip(']')
    visits_by_day_str_parsed = visits_by_day_str_parsed.dropna()
    s_visits_by_day = pd.Series([np.array(x.split(',')).astype(int) for x in visits_by_day_str_parsed], name='visits_by_day')  # converting to integer
    df_visits_by_day = s_visits_by_day.to_frame()
    df_visits_by_day.index = visits_by_day_str_parsed.index
    del visits_by_day_str, visits_by_day_str_parsed, s_visits_by_day

    # pulling monthly raw visit and visitor counts
    df_visitors_by_month = df_mobility['raw_visitor_counts'].dropna().astype(int)
    df_visits_by_month = df_mobility['raw_visit_counts'].dropna().astype(int)

    # pulling poi cbg and county IDs
    poi_cbg = df_mobility['poi_cbg'].dropna().astype(int).astype(str)
    poi_cbg[poi_cbg.str.len() < 12] = '0' + poi_cbg[poi_cbg.str.len() < 12]  # adding on stripped leading zeroes
    poi_county = poi_cbg.str.slice(0, 5).astype(int)
    poi_county = poi_county.rename('poi_county')

    # pulling and preprocessing poi monthly visitor home cbgs
    poi_monthly_visitor_home_cbg_str = df_mobility['visitor_home_cbgs'].astype('string')
    poi_monthly_visitor_home_cbg_str = poi_monthly_visitor_home_cbg_str.dropna()
    poi_monthly_visitor_home_cbg_str = poi_monthly_visitor_home_cbg_str.str.replace(':4,', ':3,')  # replacing counts of 4 with three since a value of 4 could represent 2, 3, or 4
    poi_monthly_visitor_home_cbg_str = poi_monthly_visitor_home_cbg_str.str.replace(':4}', ':3}')
    poi_monthly_visitor_home_cbg_lookup = poi_monthly_visitor_home_cbg_str.apply(lambda row: eval(row))
    poi_monthly_visitor_home_cbg_lookup = poi_monthly_visitor_home_cbg_lookup.rename('dict')
    poi_monthly_visitor_home_cbg = poi_monthly_visitor_home_cbg_lookup.apply(lambda row: list(row.keys()))
    poi_monthly_visitor_home_cbg = poi_monthly_visitor_home_cbg.rename('poi_monthly_visitor_home_cbg')
    del poi_monthly_visitor_home_cbg_str

    # calculating unassigned visits
    total_visits_from_ided_cbg = poi_monthly_visitor_home_cbg_lookup.apply(lambda row: sum(row.values()))
    poi_county_factor_weight = df_visitors_by_month - total_visits_from_ided_cbg  # assinging visits outside of number identified by visitor cbg to the county level
    poi_county_factor_weight[poi_county_factor_weight < 0] = 0

    # pulling poi county factor
    poi_county_factor = poi_county.apply(lambda row: county_factor_lookup.get(row))


    #%% Analysis

    df_poi_monthly_visitor_home_cbg = pd.DataFrame(poi_monthly_visitor_home_cbg)

    # core processing sequence > exploding out visitor home cbg to calculte appropriate factors
    visits_from_ided_cbg = df_poi_monthly_visitor_home_cbg.apply(lambda row: np.array(list(map(poi_monthly_visitor_home_cbg_lookup[row.name].get, row[0]))), axis=1)
    ided_cbg_factor = df_poi_monthly_visitor_home_cbg.apply(lambda row: np.array(list(map(cbg_month_factors_lookup.get, row[0]))), axis=1)

    # handling bad cbg match cases
    ided_cbg_is_none = ided_cbg_factor.apply(lambda row: np.isnan(row.astype(float)).any())
    df_ided_cbg_is_none = pd.DataFrame(ided_cbg_factor[ided_cbg_is_none])
    for index, row in df_ided_cbg_is_none.iterrows():
        visits_from_ided_cbg[index] = visits_from_ided_cbg[index][ided_cbg_factor[index] != None]
        ided_cbg_factor[index] = ided_cbg_factor[index][ided_cbg_factor[index] != None]

    # calculating poi monthly average scaling factor
    scaled_visits_from_ided_cbg = visits_from_ided_cbg * ided_cbg_factor
    poi_monthly_factor = (scaled_visits_from_ided_cbg.apply(np.sum, axis=0) + poi_county_factor_weight * poi_county_factor) / df_visits_by_month
    poi_monthly_factor = poi_monthly_factor.rename('factors')

    # applying scaling factor
    df_visits_by_day_and_factor = pd.concat([df_visits_by_day, poi_monthly_factor], axis=1)
    poi_scaled_visits_by_day = df_visits_by_day_and_factor['visits_by_day'] * df_visits_by_day_and_factor['factors']


    #%% File Output
    # setting up xarray
    placekey = df_mobility[df_mobility.index.isin(poi_scaled_visits_by_day.index)]['placekey']
    poi_scaled_visits_by_day_array = np.empty([np.shape(poi_scaled_visits_by_day)[0], days_in_month])
    poi_scaled_visits_by_day_array[:] = np.nan
    df_poi_scaled_visits_by_day = pd.DataFrame(poi_scaled_visits_by_day)

    c = 0
    for index, row in df_poi_scaled_visits_by_day.iterrows():
        poi_scaled_visits_by_day_array[c] = row[0]
        c += 1

    # saving netcdf of scaled visits by day
    poi_scaled_visits_by_day_ds = xr.DataArray(poi_scaled_visits_by_day_array, coords=[placekey, date_range], dims=["placekey", "date"])
    poi_scaled_visits_by_day_ds.to_netcdf('/home/rdh0715/weather_and_mobility/' + fname + '.nc')
    # poi_scaled_visits_by_day_ds.to_netcdf('/Users/ryanharp/Documents/Weather_and_Mobility/' + fname + '.nc')

    # elapsed = time.time() - t
    # print(elapsed)

    # saving corresponding csv with relevant metadata
    # placekey = df_mobility[df_mobility.index.isin(poi_scaled_visits_by_day.index)]['placekey']
    naics_code = df_mobility[df_mobility.index.isin(poi_scaled_visits_by_day.index)]['naics_code']
    latitude = df_mobility[df_mobility.index.isin(poi_scaled_visits_by_day.index)]['latitude']
    longitude = df_mobility[df_mobility.index.isin(poi_scaled_visits_by_day.index)]['longitude']
    city = df_mobility[df_mobility.index.isin(poi_scaled_visits_by_day.index)]['city']
    state = df_mobility[df_mobility.index.isin(poi_scaled_visits_by_day.index)]['region']


    poi_metadata = pd.DataFrame({'placekey': [], 'naics_code': [], 'cbg': [],
                                 'county': [], 'city': [], 'state': [], 'latitude': [], 'longitude': []})
    poi_metadata['placekey'] = placekey
    poi_metadata['naics_code'] = naics_code.astype(int)
    poi_metadata['cbg'] = poi_cbg
    poi_metadata['county'] = poi_county
    poi_metadata['city'] = city
    poi_metadata['state'] = state
    poi_metadata['latitude'] = latitude
    poi_metadata['longitude'] = longitude

    # poi_metadata.to_csv('/home/rdh0715/weather_and_mobility/' + fname + '.csv')
    poi_metadata.to_csv('/Users/ryanharp/Documents/Weather_and_Mobility/' + fname + '.csv')


#%% Execution of the Script
t = time.time()

if __name__ == '__main__':
    pool = mp.Pool(processes=9)
    pool.map(wrapper, ['2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
                    '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12',
                    '2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06',
                    '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12',
                    '2020-01', '2020-02'])

elapsed = time.time() - t
print(elapsed)
