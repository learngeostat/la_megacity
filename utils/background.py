#!/usr/bin/env python
"""
The main purpose of the functions in this estimate background concentrations of CO2 and CH4 in
Los Angeles Basin. It first reads raw concentration data from the files. This task is accomplished
through function compute_background_filters. THis is followed by filtering the data to remove observations
that high hourly standard deviation and that have high hour to hour variability. Only observations
below certain sd and hour to hour variability in concentrations are retained.

For details regarding these criteria read
Verhulst, K. R., Karion, A., Kim, J., Salameh, P. K., Keeling, R. F., Newman, S., Miller, J., Sloop, C., Pongetti, T.,
Rao, P., Wong, C., Hopkins, F. M., Yadav, V., Weiss, R. F., Duren, R. M., and Miller, C. E.: Carbon dioxide and
methane measurements from the Los Angeles Megacity Carbon Project – Part 1: calibration, urban enhancements,
and uncertainty estimates, Atmos. Chem. Phys., 17, 8313–8341, https://doi.org/10.5194/acp-17-8313-2017, 2017.
Specific algorithm details appear in the supplementary material for the paper
https://acp.copernicus.org/articles/17/8313/2017/acp-17-8313-2017-supplement.pdf

Once all the filters have been applied CCGCRV curve fitting method is used to get smoothed estimate of the background

"""
import copy
import re

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Optional, Dict, Union


from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec



from la_megacity.utils import conc_func as cfunc 
from la_megacity.utils import constants as prm
from la_megacity.utils.CCGRV import ccgfilt

__author__ = "Vineet Yadav"
__copyright__ = "Copyright 2023, The Megacity Project"
__version__ = "1"
__maintainer__ = "Vineet Yadav"
__email__ = "vineet.yadav@jpl.nasa.gov"
__status__ = "Production"


# %% Add decimal dates to dataframe
def add_decimal_dates(site_data: pandas.DataFrame) -> pandas.DataFrame:
    """
    The main goal of this function is to take a dataframe consisting of column datetime_UTC that has time
    in format yyyy-MM-dd hh:mm:ss and convert them to decimal_dates and insert it after 6th column or at sixth position
    based 0 based indexing
    :param site_data: a dataframe that consists of datetime_UTC column as well as year, month, day, hour, minute,
    numeric time in separate columns among other information.
    :return:
    site_data an input dataframe that has an extra column consisting of decimal dates
    """
    # Convert to list for conversion of dates to decimal dates. These decimal dates are required by CCGCRV
    # for creating smooth curves
    time_decimal = site_data['datetime_UTC'].tolist()
    # Convert to decimal date
    time_decimal = cfunc.toYearFraction(time_decimal).flatten()
    # Insert decimal time in dataframe at a specific position that is sixth column
    site_data.insert(6, "decimal_time", time_decimal)
    return site_data


# %% Return consecutive counts of 1's in a long string of 0 and 1
def return_string_occurrence(string_select_condition: str, num_hours: float, match_indices_required: bool) -> \
        tuple[numpy.array, list]:
    """
    The main goal of this function is to take a long string of 0 and 1 and identify occurrence of 1's in that string.
    This consists of identifying the beginning and ending index of 1's and the count of consecutive ones. This
    information can then be used to filter observations that fulfill consecutive hour requirdement. Location of 1's are
    the observations that pass the consecutive hour filter
    :type num_hours: float
    :type string_select_condition: str
    :type match_indices_required: bool
    :param string_select_condition: A string consisting of 0 and 1.
    :param num_hours: a float value which would be used to filter sub-strings by consecutive hour requirement
    and match indices for those sub-strings i.e., begin and ending index of sub-strings that pass the
    consecutive hour requirement are returned .
    :param match_indices_required: This is a bool that is True or False. If this is set to true then the function
    would return the range of indices of sub-strings of ones. For e.g. returned as a list [ 23, 30 , 7] i.e. sub-string
    between 23rd and 30th position of length 7 consists of all ones.
    :return:
    index_store : a numpy array consisting of zeros and ones. It is only filled with ones which occur consecutively for
    >= num_hours
    match_index : a list of lists : range of sub-string of 1's in the string
    """
    index_store = np.zeros(len(string_select_condition), dtype=int)

    if match_indices_required:
        match_index = []

    for match in re.finditer(pattern='1+', string=string_select_condition):
        if match.end() - match.start() >= num_hours:
            if match_indices_required:
                match_index.append([match.start(), match.end(), match.end() - match.start()])
            index_store[match.start():match.end()] = 1

    if match_indices_required:
        return index_store, match_index
    else:
        return index_store


# %% Generate CO2 SD and Stability Filters
def stability_std_filter(initial_site_data: pandas.DataFrame, column_name_gas: str, column_name_gas_std: str,
                         gas_stability_filter: float, gas_sd_filter: float) -> pandas.DataFrame:
    """
    The main goal of this function is to identify which carbon dioxide observations pass standard deviation and
    stability criteria
    :rtype: pandas.DataFrame
    :param initial_site_data: (pandas dataframe): This is the carbon dioxide data initially stored in the files provided
    for individual sites. This variable is not modified by the function.
    :param column_name_gas: a string: Name of the column that contains carbon dioxide measurement values
    :param column_name_gas_std: a string: Name of the column that contains carbon dioxide measurement hourly
    standard deviation
    :param gas_stability_filter: a float: This is value of stability of changes in hourly to hourly variations
    of carbon dioxide measurements in ppm
    :param gas_sd_filter: a float: This is value which is used to indication which carbon dioxide measurements pass
    hourly standard deviation criteria (units ppm)
    :return: On return a variable site_data_filled that contains indicator variables indicating which carbon dioxide
    observations pass stability and standard deviation criteria.
    """
    # For CO2 both Standard Deviation and Stability Filter is required. Here we are doing filtering operation
    # conditions that is just creating column of 0 and 1 satisfying those conditions'
    # step 1
    # Stability filter retain measurements where hour to hour difference is less than gas_stability_filter value in ppm

    # compute derivative or hourly differences between carbon dioxide observations. Append NaN as derivative returns
    # values that are of length 1 < length of original measurements
    stability_query = np.append(np.nan, np.abs(np.diff(initial_site_data[column_name_gas])))
    # get the gas name by splitting string column gas name
    gas_name_split = column_name_gas.split("_")
    # filter based on gas_stability_filter
    condition_stability = np.where(stability_query <= gas_stability_filter)  # np.where returns a tuple
    condition_stability = condition_stability[0]  # extract indices from a tuple
    initial_site_data = initial_site_data.assign(gas_stability_condition=0)  # add a column for stability indicator and
    # initialize with zeros
    initial_site_data.iloc[condition_stability, -1] = 1  # add ones for observation that match stability criteria
    # add a column that has stability values or hour to hour differences
    initial_site_data = initial_site_data.assign(gas_stability_val=stability_query)
    # step 2
    # SD condition retain measurements where standard deviation is less than  equal to standard deviation filter
    sd_condition = column_name_gas_std + "<=" + str(gas_sd_filter)
    sd_filter_index = \
        initial_site_data.index.get_indexer(initial_site_data.query(sd_condition).index)
    # initialize indicator variable with zeros
    initial_site_data = initial_site_data.assign(gas_SD_condition=0)
    # For observations that pass SD conditions replace zeros by ones
    initial_site_data.iloc[sd_filter_index, -1] = 1
    # Name of columns by gas type
    stability_condition_name = gas_name_split[0] + '_stability_condition'
    sd_condition_name = gas_name_split[0] + '_SD_condition'
    # Rename columns
    initial_site_data = initial_site_data.rename(columns={"gas_stability_condition": stability_condition_name})
    initial_site_data = initial_site_data.rename(columns={"gas_SD_condition": sd_condition_name})
    data_filled = initial_site_data
    return data_filled


# %% Generate CH4 SD filter
def std_filter(initial_data_site: pandas.DataFrame, site_data_filled: pandas.DataFrame,
               gas_sd_filter: float, column_name_gas: str, column_name_gas_std: str) -> pandas.DataFrame:
    """
    The main goal of this function is to identify which methane observations pass standard deviation criteria
    :param initial_data_site: (pandas dataframe): This is the CH4 data initially stored in the files provided
    for individual sites. This variable is not modified by the function
    :param site_data_filled: (pandas dataframe): This is a dataframe that contains CO2 measurements at 1 minute interval
    and its standard deviation and contains filtered indicator variables for filtering CO2 values based on standard
    deviation filter and hourly to hourly difference filter. This dataframe is modified by this function and on return
    contains an indicator variable based on standard deviation criteria for methane or ch4
    :param gas_sd_filter: a float: Standard deviation filter value for methane or ch4
    :param column_name_gas: a string: Name of the column that contains methane measurement values
    :param column_name_gas_std: a string: Name of the column that contains methane measurement hourly standard deviation
    :return: modified dataframe site_data_filled that contains indicator of 0 and 1 called as ch4_SD_condition that
    tells which measurement passes the CH4 standard deviation criteria
    """
    # As the name implies the function creates an indicator variable of 0 and 1 identifying which methane observations
    # pass standard deviation criteria
    # form a query string to obtain values
    sd_condition = column_name_gas_std + "<=" + str(gas_sd_filter)
    # get result after filtering
    sd_filter_index = initial_data_site.index.get_indexer(initial_data_site.query(sd_condition).index)
    # in the site_data_filled dataframe add methane observations stored in initial_data_site dataframe
    site_data_filled[column_name_gas] = initial_data_site[column_name_gas]
    # in the site_data_filled dataframe add hourly methane standard deviation stored in initial_data_site dataframe
    site_data_filled[column_name_gas_std] = initial_data_site[column_name_gas_std]
    # split the text to construct column name
    sd_name_split = column_name_gas_std.split("_")
    construct_name = sd_name_split[0] + '_' + sd_name_split[1] + '_' + 'condition'
    # Initialize an indicator column with zeros
    site_data_filled = site_data_filled.assign(gas_SD_condition=0)
    # Replace zeros in initialized column in the previous step by 1 for filtered methane observations based on SD
    # criteria
    site_data_filled.iloc[sd_filter_index, -1] = 1
    # Rename the column to match exact gas use construct gas
    site_data_filled = site_data_filled.rename(columns={"gas_SD_condition": construct_name})
    return site_data_filled


# %% Combine Multiple Filters to Filter Observations CO2 and CH4 observations
# Two SD filters & 1 stability filter & 1 consecutive hour filter
def combine_all_filters(site_data_filled: pandas.DataFrame, num_hours: float, match_indices_required: bool) \
        -> tuple[pandas.DataFrame, pandas.DataFrame, list]:
    """
    The main goal of this function is to combine multiple filters to create an indicator variable that tells which
    carbon dioxide observations pass all criteria. Indicator variables for first two filters mentioned below are
    created by the function stability_std_filter. Indicator variable with respect to consecutive hour criteria
    that is third criteria mentioned below is determined in this function and furthermore it is combined with other
    two criteria to create a final indicator variable that identifies observations which pass all three criteria
    (1) Combined CO2 and CH4 SD criteria
    (2) CO2 stability criteria (hour to hour differences)
    (3) CO2 multiple hour stability criteria (num_hours = 6)

    :param site_data_filled: a dataframe (modified by the function) : When entering the function the site_data_filled
    dataframe has all indicator variables associated with first two criteria. A third indicator variable is added for
    the consecutive hour requirement. All these three criteria are combined to create and overall indicator variable
    that indicates whether an observation has passed all three criteria mentioned above.
    :param num_hours: a float value which would be used to filter observation by consecutive hour requirement
    :param match_indices_required: This is a bool that is True or False. If this is set to true then the function
    would return the range of indices (row numbers) of indicator variable that are part of overall indicator variable
    For e.g. returned as a list [ 23, 30 , 7] i.e. 23rd to 30th observation pass (that is 7 observations) pass all three
    filters
    :return:
    selected_data : a dataframe : This is a dataframe only consisting of observations that pass all three criteria
    site_data_filled : a dataframe : This is a dataframe that has all observations and all indicator variables that
    can be used to filter observations. Selected_data is a dataframe obtained after applying all three filters
    match_index : a list of lists : range of observations that fulfill all criteria
    """
    try:
        # First attempt - CO2 and CH4
        condition_selection = site_data_filled['co2_stability_condition'] + site_data_filled['co2_SD_condition'] + \
                             site_data_filled['ch4_SD_condition']
    except (KeyError, Exception) as e:
        try:
            # Second attempt - CO
            condition_selection = site_data_filled['co2_stability_condition'] + site_data_filled['co_SD_condition'] + \
                                site_data_filled['co_SD_condition']
        except (KeyError, Exception) as e:
            print(f"Error: Could not create condition selection. {str(e)}")
            # You might want to handle this case, perhaps by setting a default or raising an error
            raise
    condition_selection = condition_selection.to_numpy()
    # find observations that fulfill stability and sd criteria
    condition_selection = (condition_selection == 3).astype(int)  # 2 SD condition and 1 stability condition total 3
    # Create an indicator variable in site_data_filled dataframe to indicate which observations fulfill the
    # stability and SD criteria
    site_data_filled = site_data_filled.assign(co2_obs_selection=condition_selection)
    # concatenate 0 and 1 from variable co2_obs_selection to create a string of 0 and 1 for building
    # consecutive hours filter
    string_select_condition = ''.join(str(i) for i in site_data_filled['co2_obs_selection'].tolist())
    # check if indexes of observations that fulfill all criteria are required
    if match_indices_required:
        index_store, match_index = return_string_occurrence(string_select_condition, num_hours, match_indices_required)
    else:
        index_store = return_string_occurrence(string_select_condition, num_hours, match_indices_required)

    # Add indicator variable for observations that pass all three criteria
    site_data_filled['final_obs_selection'] = index_store.tolist()

    # create dataframe of observations that pass all criteria
    selected_data = site_data_filled.loc[site_data_filled['final_obs_selection'] == 1]

    if match_indices_required:
        return selected_data, site_data_filled, match_index
    else:
        match_index = []
        return selected_data, site_data_filled, match_index


# %% Read files that store co2 and ch4 concentration data and call two filter functions to identify observations
# that pass all filters
def compute_background_filters(path_for_files: str, gas_type: str,
                               meas_units: list, begin_year: int, end_year: int,
                               time_frequency: str, site_name: str, inlet_height: str) -> pandas.DataFrame:
    """
    The main goal of this function is to compute SD and Stability filters and obtained filtered observations
    for a site for computing background. Also see parameters.py.
    This is a wrapper function that other than calling appropriate functions for computing SD and Stability indicator
    filters also performs an important function of reading data from concentration data stored in text files

    :param path_for_files: path where csv files of concentrations are stored
    :param gas_type: a string listing gas types for which background would be computed like ['co2','ch4']
    Note. Both co2 and ch4 concentrations are required for computing co2 and ch4 background
    :param meas_units: a list measurement units of each gas included in list gas_type
    :param begin_year: an int: year from which observations are available
    :param end_year: an int: year until which observations are available
    :param time_frequency: a string: indicating frequency at which observations are available. For e.g. 'H' for hourly
    :param site_name: name of the site from which background concentrations would be computed
    :param inlet_height: inlet height of the instrument located at the site.
    :param gas_stability_filter: a list of floats indicating in ppm or ppb the hourly to hourly variations in
    concentration measurements below which observations would be included in background calculation. In Kris's paper
    only mentioned for CO2
    :param gas_sd_filter: a list of floats indicating in ppm or ppb maximal SD of observations allowed within an hour
    :return:
    site_data_filled: a dataframe consisting of SD and stability filters (indicator variables) associated with different
    gas types for a site (Note primarily CO2 and CH4)
    """
    if gas_type in ['co2', 'ch4']:
         gas_identity = ['co2', 'ch4']
         meas_units = ['ppm', 'ppb']
    elif gas_type == 'co':
         gas_identity = ['co2','co']
         meas_units = ['ppm','ppb']

    total_gases = len(gas_identity)

    for j in range(total_gases):
        # Read files for co2 and ch4 for a site
        data_site = cfunc.collect_allsite_data(path_for_files, gas_identity[j],
                                               meas_units[j], begin_year, end_year,
                                               time_frequency, [site_name], [inlet_height])
        # Name of Standard Deviation Column
        gas_std_column_name = gas_identity[j] + '_SD_minutes_' + site_name
        # Name of Gas Column
        gas_column_name = gas_identity[j] + '_' + meas_units[j] + '_' + site_name
        # Stability & SD Criteria from Kris Paper for different gases
        # This is just collecting data for analysis
        if gas_identity[j] == 'co2':
            site_data_filled_co2 = stability_std_filter(data_site, gas_column_name, gas_std_column_name,
                                                        prm.gas_stability_filter[site_name][gas_identity[j]],
                                                        prm.gas_sd_filter[site_name][gas_identity[j]]) #prm.gas_sd_filter[gas_identity[j]]

        elif gas_identity[j] == 'ch4':
            site_data_filled = std_filter(data_site, site_data_filled_co2,
                                          prm.gas_sd_filter[site_name][gas_identity[j]], gas_column_name, #prm.gas_sd_filter[gas_identity[j]]
                                          gas_std_column_name)
        # co functionality is built but not implemented
        elif gas_identity[j] == 'co':
            site_data_filled = std_filter(data_site, site_data_filled_co2,
                                          prm.gas_sd_filter[site_name][gas_identity[j]], gas_column_name, 
                                          gas_std_column_name)
    a=1
    return site_data_filled


# %% Compute Background for a chosen gas
def background_gas_type(site_data_filled: pandas.DataFrame, gas_type: str, num_hours: float, match_indices: bool) \
        -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    The main goal of this function is to combine multiple filters for a gas and extract observations that can be passed
    to CCGCRV program for computing smooth curve.
    :param site_data_filled: a dataframe consisting of SD and stability filters. THis can be obtained by using function
    compute_background_filters.
    :param gas_type: a string listing gas types for which background would be computed like 'co2'
    :param num_hours: num_hours: a float value which would be used to filter observation by consecutive hour requirement
    :param match_indices: This is a bool that is True or False. If this is set to true then the function
    would return the range of indices (row numbers) for filtered observations.
    For e.g. returned as a list [ 23, 30 , 7] i.e. 23rd to 30th observation pass (that is 7 observations) pass all
    criteria and should be used in CCGCRV for computing a smooth curve.
    :return:
    site_data_filled: a dataframe consisting of all observations and all filters with a column of decimal dates
    selected_data: a dataframe consisting of selected observations and all filters with a column of decimal dates

    """
    if gas_type == 'co2':
        selected_data, site_data_filled, _ = combine_all_filters(site_data_filled, num_hours, match_indices)
    elif gas_type == 'ch4':
        selected_data, site_data_filled, _ = combine_all_filters(site_data_filled, num_hours, match_indices)
    elif gas_type == 'co':
        # use ch4 filters for now
        selected_data, site_data_filled, _ = combine_all_filters(site_data_filled, num_hours, match_indices)

    site_data_filled = add_decimal_dates(site_data_filled)
    selected_data = add_decimal_dates(selected_data)

    return selected_data, site_data_filled


# %% Extract all statistics from the CCGCRV call
def get_ccgcrv_stat(ccgcrv_object: ccgfilt.ccgFilter) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """
    The main goal of this function is to take object returned by the CCGCRV function and return all the statistics
    for plotting, visualization and also provide smooth curve estimate
    :param ccgcrv_object: An object obtained by calling CCGRCV class
    :return: (1) original statistics dataframe consisting various statistics at time points that are used to estimate the
    smooth curve, (2) predicted_statistics a dataframe that consists of estimates of smooth curve at temporal interval
    specified in parameter prm.sample_interval_equal, (3) a datframe of monthly means of the smooth curve
    """
    original_statistics = {'decimal_time': ccgcrv_object.xp,
                           # original data values in c code files. option -orig in ccgcrv code
                           'input_dependent_variable': ccgcrv_object.yp,
                           # gr option in c code of ccgcrv
                           'growthrate_value': ccgcrv_object.getGrowthRateValue(ccgcrv_object.xp),
                           # option harm in the c code
                           'harmonic_value': ccgcrv_object.getHarmonicValue(ccgcrv_object.xp),
                           # poly option in c code file
                           'polynomial_value': ccgcrv_object.getPolyValue(ccgcrv_object.xp),
                           # trend option in c code files
                           'trend': ccgcrv_object.getTrendValue(ccgcrv_object.xp),
                           # option func in c code file
                           'function_value': ccgcrv_object.getFunctionValue(ccgcrv_object.xp),
                           # res in c code files
                           'residual_from_function': ccgcrv_object.resid,
                           # ressm in c code files
                           'residuals_from_smooth': ccgcrv_object.yp - ccgcrv_object.getSmoothValue(ccgcrv_object.xp),
                           # detrend in c code files
                           'detrend': ccgcrv_object.yp - ccgcrv_object.getTrendValue(ccgcrv_object.xp),
                           # smcycle option in c code files
                           'smooth_detrended_cycle': ccgcrv_object.getSmoothValue(ccgcrv_object.xp) -
                                                     ccgcrv_object.getTrendValue(ccgcrv_object.xp),
                           # value of the smooth curve at original location
                           'smooth_value': ccgcrv_object.getSmoothValue(ccgcrv_object.xp),
                           }
    original_statistics = pd.DataFrame.from_dict(original_statistics)
    date_vector = cfunc.fraction_date_todatetime(ccgcrv_object.xp)
    original_statistics = original_statistics.join(date_vector)

    predicted_statistics = {'decimal_time': ccgcrv_object.xinterp,
                            'growthrate_value': ccgcrv_object.getGrowthRateValue(ccgcrv_object.xinterp),
                            'harmonic_value': ccgcrv_object.getHarmonicValue(ccgcrv_object.xinterp),
                            'polynomial_value': ccgcrv_object.getPolyValue(ccgcrv_object.xinterp),
                            'trend': ccgcrv_object.getTrendValue(ccgcrv_object.xinterp),
                            'function_value': ccgcrv_object.getFunctionValue(ccgcrv_object.xinterp),
                            'smooth_value': ccgcrv_object.getSmoothValue(ccgcrv_object.xinterp),
                            }
    predicted_statistics = pd.DataFrame.from_dict(predicted_statistics)
    date_vector = cfunc.fraction_date_todatetime(ccgcrv_object.xinterp)
    predicted_statistics = predicted_statistics.join(date_vector)

    # Overall Statistics Monthly Means
    monthly_means = ccgcrv_object.getMonthlyMeans()
    year_monthly_means = [x[0] for x in monthly_means]
    month_monthly_means = [x[1] for x in monthly_means]
    estimates_monthly_means = [x[2] for x in monthly_means]
    combine_data = {'year': year_monthly_means, 'month': month_monthly_means, 'estimates': estimates_monthly_means}
    monthly_means = pd.DataFrame(data=combine_data)  # As a dataframe

    return original_statistics, predicted_statistics, monthly_means


# %% CCGCRV Call
def ccgcrv(site_name: str, gas_type: str, meas_units: str, data_for_smoothing: pandas.DataFrame) -> ccgfilt.ccgFilter:
    """
    :param site_name: The name of the site that is being used for computing the background
    :param gas_type: a string: indicating the gas for which background is being computed
    :param meas_units: a string: indicating the measurement units for the gas
    :param data_for_smoothing: a dataframe consisting of selected observations that would be used for obtaining smoothed
    curve from CCGCRV. This dataframe can be obtained by calling in sequence functions
    compute_background_filters
    background_gas_type
    :return:
    an object filt_results obtained by calling CCGCRV function that consists of all results
    """
    gas_column_name = gas_type + '_' + meas_units + '_' + site_name
    dependent_variable = data_for_smoothing[gas_column_name].to_numpy()
    independent_variable = data_for_smoothing['decimal_time'].to_numpy()  # decimal_time
    time_zero = independent_variable[0]
    filtered_results = ccgfilt.ccgFilter(independent_variable,
                                         dependent_variable,
                                         prm.short_term,
                                         prm.long_term,
                                         prm.sample_interval_equal,
                                         prm.polynomial_terms,
                                         prm.numer_harmonics,
                                         time_zero,
                                         prm.gap,
                                         prm.debug)

    original_statistics, predicted_statistics, monthly_means = get_ccgcrv_stat(filtered_results)

    return original_statistics, predicted_statistics, monthly_means, filtered_results


# %% CCGCRV Call
def ccgcrv_run_results(independent_variable: numpy.array, dependent_variable: numpy.array, time_zero: float) -> \
        tuple[pandas.DataFrame, pandas.DataFrame, ccgfilt.ccgFilter]:
    """
    The main goal of this function is to run CCGCRV by utilizing independent (gas concentrations)  and dependent
    variable (decimal time) to estimate various statistics at original time points and at equal temporal interval
    specified in the parameter prm.sample_interval_equal. The only difference between this function and ccgcrv function
    is that this one takes input an independent and dependent variable and time_zero i.e., the beginning time of the
    first value and the other one constructs the column name from which to extract dependent variable based on the gas
    name and return the results in CCGCRV object
    :param independent_variable: time in decimal yearly format
    :param dependent_variable: gas concentrations or field to be smoothed
    :param time_zero: time associated with first value
    :return: (1) original statistics dataframe consisting various statistics at time points that are used to estimate
    th smooth curve, (2) predicted_statistics a dataframe that consists of estimates of smooth curve at temporal
    interval specified in parameter prm.sample_interval_equal, (3) an object obtained through call to CCGCRV function
    for assessment.
    """
    filt_results = ccgfilt.ccgFilter(independent_variable,
                                     dependent_variable,
                                     prm.short_term,
                                     prm.long_term,
                                     prm.sample_interval_equal,
                                     prm.polynomial_terms,
                                     prm.numer_harmonics,
                                     time_zero,
                                     prm.gap,
                                     prm.debug)
    original_statistics, predicted_statistics, _ = get_ccgcrv_stat(filt_results)

    return original_statistics, predicted_statistics, filt_results


# %% Apply filters based smooth curve fit
def ccgcrv_large_residual_removal(site_name: str, gas_type: str, data_for_smoothing: pandas.DataFrame) -> \
        tuple[bool, pandas.DataFrame]:
    """
    :param site_name: The name of the site that is being used for computing the background
    :param gas_type: a string: indicating the gas for which background is being computed
    :param data_for_smoothing: a dataframe consisting of selected observations that would be used for obtaining smoothed
    curve from CCGCRV. This dataframe can be obtained by calling in sequence functions
    compute_background_filters
    background_gas_type
    """
    # Note CO2 and CH4 has to be done simultaneously as they have same filtering criteria
    if gas_type == 'co2' or 'ch4':
        gas_identity = ['co2', 'ch4']
        meas_units = ['ppm', 'ppb']

        independent_variable = data_for_smoothing['decimal_time'].to_numpy()  # decimal_time
        time_zero = independent_variable[0]

        # Extract Dependent CO2 data
        gas_column_name_co2 = gas_identity[0] + '_' + meas_units[0] + '_' + site_name
        dependent_variable = data_for_smoothing[gas_column_name_co2].to_numpy()

        # Run CCGCRV for co2 data
        original_stat_co2, pred_stat_co2, gas_filtered_co2 = ccgcrv_run_results(independent_variable,
                                                                                dependent_variable,
                                                                                time_zero)

        # get standard deviation of smooth cycle at original points (not at interpolated equidistant points).
        # These estimates are obtained through function getFunctionValue in class ccgfilt
        std_smooth_values = np.std(original_stat_co2['smooth_value'])
        # find large residuals from smooth curve
        large_residuals_co2 = original_stat_co2['residuals_from_smooth'] <= 2 * std_smooth_values
        # convert True and False bool to 0 and 1 int value
        large_residuals_co2 = large_residuals_co2.to_numpy().astype(int)

        # Extract Dependent CH4 data
        gas_column_name_ch4 = gas_identity[1] + '_' + meas_units[1] + '_' + site_name
        dependent_variable = data_for_smoothing[gas_column_name_ch4].to_numpy()

        # Run CCGCRV for ch4 data
        original_stat_ch4, pred_stat_ch4, gas_filtered_ch4 = ccgcrv_run_results(independent_variable,
                                                                                dependent_variable, time_zero)
        # get standard deviation of smooth cycle at original points (not at interpolated equidistant points).
        # These estimates are obtained through function getFunctionValue in class ccgfilt
        std_smooth_values = np.std(original_stat_ch4['smooth_value'])
        # find large residuals from smooth curve
        large_residuals_ch4 = original_stat_ch4['residuals_from_smooth'] <= 2 * std_smooth_values
        # convert True and False bool to 0 and 1 int value
        large_residuals_ch4 = large_residuals_ch4.to_numpy().astype(int)

        # Add large residual condition
        # This should be at max 2, which indicates the observations pass both standard deviation criteria
        check_condition = large_residuals_ch4 + large_residuals_co2

        # if all observations are retained then we just return the original dataframe and indicate by bool True that
        # iteratively filtered observation has been obtained
        if sum(check_condition) == len(large_residuals_co2) + len(large_residuals_ch4):  # total obs
            # are sum(check_condition)/2
            print('Total Observations : ' + str(len(check_condition)) + ' and ' + str(0) + ' large residuals found.')
            observations_not_deleted = True
            return observations_not_deleted, data_for_smoothing
        # When all observations are not retained then extract observations that pass the filter and return False
        # indicating that some observations have been deleted.
        else:
            large_residuals_index = np.where(check_condition == 2)  # np.where returns a tuple
            large_residuals_index = large_residuals_index[0]  # extract indices from a tuple
            data_for_smoothing = data_for_smoothing.iloc[large_residuals_index]
            print('Total Observations : ' + str(len(check_condition)) + ' and ' + \
                  str(len(check_condition) - len(large_residuals_index)) + ' large residuals found.')
            observations_not_deleted = False
            return observations_not_deleted, data_for_smoothing

    elif gas_type == 'co':
        gas_identity = ['co']
        meas_units = ['ppb']
        print('Functionality Not Built for CO')
        observations_not_deleted = True
        return observations_not_deleted, data_for_smoothing


# %% Final Background Function Run at max iterations
def background_return(site_name: str, gas_type: str, meas_units: str, data_for_smoothing: pandas.DataFrame) -> \
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, ccgfilt.ccgFilter]:
    """
    The main goal of this function is to iteratively call ccgcrv function to remove observations with large residual
    and recompute smoothed curve until all observations are below 2SD of smooth curve. Note a maximum of 7 iterations
    are allowed but in most situations no more than 1 or 2 are required
    See function ccgcrv_large_residual_removal for details
    :param site_name: The name of the site that is being used for computing the background
    :param gas_type: a string: indicating the gas for which background is being computed
    :param meas_units: a string: indicating the measurement units for the gas
    :param data_for_smoothing: a dataframe consisting of selected observations that would be used for obtaining smoothed
    curve from CCGCRV.
    :return: (1) original statistics dataframe consisting various statistics at time points that are used to estimate
    the smooth curve, (2) predicted_statistics a dataframe that consists of estimates of smooth curve at temporal
    interval specified in parameter prm.sample_interval_equal, (3) a datframe of monthly means of the smooth curve
    """
    total_iterations = 7
    # for i in range(total_iterations):
    #     condition, data_for_background = ccgcrv_large_residual_removal(site_name, gas_type, data_for_smoothing)
    #     if condition:
    #         break
    #     else:
    #         data_for_smoothing = data_for_background
    condition = False
    while not condition:
        condition, data_for_background = ccgcrv_large_residual_removal(site_name, gas_type, data_for_smoothing)
        data_for_smoothing = data_for_background

    _, _, _, background_ccgcrv_result = ccgcrv(site_name, gas_type, meas_units, data_for_smoothing)
    original_stat, predicted_stat, monthly_means = get_ccgcrv_stat(background_ccgcrv_result)
    return original_stat, predicted_stat, monthly_means, background_ccgcrv_result

#%%
def subtract_smooth_background(all_data, smooth_background, sites, gas_type, units):
    """
    Subtract smooth background from gas data using exact time match.
    
    Parameters:
    -----------
    all_data : pd.DataFrame
        Contains gas data with 'datetime_UTC' and '{gas_type}_{units}_{site}'.
    smooth_background : pd.DataFrame
        Contains background with 'datetime_UTC' and 'background'.
    sites : list
        List of site names.
    gas_type : str
        Type of gas (e.g., 'co2', 'ch4')
    units : str
        Units of measurement (e.g., 'ppm', 'ppb')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with background subtracted from each site's measurements
    """
    # Generate required column names
    required_cols = ['datetime_UTC']
    site_cols = [f'{gas_type}_{units}_{site}' for site in sites]
    required_cols.extend(site_cols)
    
    # Validate columns in input DataFrames
    if not all(col in all_data.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in all_data.columns]
        raise ValueError(f"Missing required columns in all_data: {missing_cols}")
    if not all(col in smooth_background.columns for col in ['datetime_UTC', 'background']):
        raise ValueError("Missing columns in smooth_background: datetime_UTC, background")
    
    # Create copy of input data and merge with background
    result_data = all_data.copy()
    result_data = pd.merge(result_data, 
                          smooth_background[['datetime_UTC', 'background']], 
                          on='datetime_UTC', how='left')
    
    # Subtract background from each site's measurements
    for col_name in site_cols:
        result_data[col_name] -= result_data['background']
    
    # Remove background column and return
    return result_data.drop('background', axis=1)
#%% Combine backgrounds
def combine_backgrounds(dfA, dfB, begin_year, end_year):
   """Combine two background dataframes using timeline from larger dataframe with rounded times."""
   """ First use return_background_smoothed() to get background for VIC and SCI by removing spikes and
   then combine them. Note ccgcrv requires decimal time therefore when we convert to closest hour it leaves
   time as 11:59:59 which has been rounded to nearest hour. Note it is mean background by taking mean of smoothed              background for both sites"""
   # Initial setup and validation
   dfA = dfA.copy()
   dfB = dfB.copy()
   dfA['datetime_UTC'] = pd.to_datetime(dfA['datetime_UTC']).dt.round('h')
   dfB['datetime_UTC'] = pd.to_datetime(dfB['datetime_UTC']).dt.round('h')

   # Use timeline from larger dataframe
   base_df = dfA if len(dfA) > len(dfB) else dfB
   base_times = pd.to_datetime(base_df['datetime_UTC'])
   
   result = pd.DataFrame({
       'datetime_UTC': base_times,
       'year': base_times.dt.year,
       'month': base_times.dt.month,
       'day': base_times.dt.day,
       'hour': base_times.dt.hour,
       'minute': 0  # Always 0 after rounding to hour
   })
   
   # Recalculate numeric_time after rounding
   result['numeric_time'] = (result['datetime_UTC'] - pd.Timestamp('1970-01-01')).dt.total_seconds() / (24 * 3600)
   
   # Merge and calculate mean
   result = pd.merge(result, dfA[['datetime_UTC', 'smooth_value']], 
                    on='datetime_UTC', how='left', suffixes=('', '_A'))
   result = pd.merge(result, dfB[['datetime_UTC', 'smooth_value']], 
                    on='datetime_UTC', how='left', suffixes=('_A', '_B'))
   
   result['background'] = result[['smooth_value_A', 'smooth_value_B']].mean(axis=1, skipna=True)
   result = result.drop(['smooth_value_A', 'smooth_value_B'], axis=1)
   
   return result


# %% Print figures
def print_figures(predicted_stat: pandas.DataFrame, original_stat: pandas.DataFrame):
    estimation_time = predicted_stat['decimal_time'].to_numpy()
    smooth_value = predicted_stat['smooth_value'].to_numpy()
    trend_smooth_value = predicted_stat['trend'].to_numpy()
    input_time = original_stat['decimal_time'].to_numpy()
    input_points = original_stat['input_dependent_variable'].to_numpy()
    growth_rate = original_stat['smooth_detrended_cycle'].to_numpy()
    fig1, ax1 = plt.subplots()

    # ax1.plot(estimation_time, smooth_value, color='green', label='Smoothed Time Series')
    # ax1.plot(estimation_time, trend_smooth_value, color='red', label='Trend')
    # ax1.plot(input_time, input_points, '.', color='salmon', label='select_points')
    # ax1.plot(estimation_time, smooth_value, color='green', label='Smoothed Time Series')
    ax1.plot(input_time, growth_rate, color='green', label='growth_rate')

    plt.show()

#%%
def create_time_series_plots(
    predicted_stats: dict,
    original_stats: dict,
    figsize: Tuple[int, int] = (12, 8),
    plot_type: str = 'all',
    style: str = 'seaborn',
    dpi: int = 100
) -> Dict[str, Tuple[Figure, Union[Axes, Tuple[Axes, Axes]]]]:
    """
    Creates publication-quality time series plots with multiple components for multiple sites.
    
    Args:
        predicted_stats: Dictionary of DataFrames containing predicted statistics, keyed by site
        original_stats: Dictionary of DataFrames containing original data, keyed by site
        figsize: Tuple of figure dimensions (width, height)
        plot_type: Type of plot to generate ('all', 'smooth', 'growth', 'trend')
        style: Matplotlib style to use
        dpi: Dots per inch for the figure
    
    Returns:
        Dictionary of (Figure, Axes) tuples keyed by site
    """
    # Required columns validation
    required_predicted_cols = ['decimal_time', 'smooth_value', 'trend']
    required_original_cols = ['decimal_time', 'input_dependent_variable', 'smooth_detrended_cycle']
    
    # Set style
    plt.style.use(style)
    
    # Store figures for each site
    figures_dict = {}
    
    # Create plots for each site
    for site in predicted_stats.keys():
        predicted_stat = predicted_stats[site]
        original_stat = original_stats[site]
        
        # Validate columns for each site
        if not all(col in predicted_stat.columns for col in required_predicted_cols):
            raise ValueError(f"predicted_stat for {site} missing required columns: {required_predicted_cols}")
        if not all(col in original_stat.columns for col in required_original_cols):
            raise ValueError(f"original_stat for {site} missing required columns: {required_original_cols}")

        # Extract data
        estimation_time = predicted_stat['decimal_time'].to_numpy()
        smooth_value = predicted_stat['smooth_value'].to_numpy()
        trend_smooth_value = predicted_stat['trend'].to_numpy()
        input_time = original_stat['decimal_time'].to_numpy()
        input_points = original_stat['input_dependent_variable'].to_numpy()
        growth_rate = original_stat['smooth_detrended_cycle'].to_numpy()

        if plot_type == 'all':
            fig = plt.figure(figsize=figsize, dpi=dpi)
            gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
            
            # Top subplot for time series and trend
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(estimation_time, smooth_value, 
                     color='#2ecc71', linewidth=2, label='Smoothed Time Series')
            ax1.plot(estimation_time, trend_smooth_value, 
                     color='#e74c3c', linewidth=2, label='Trend')
            ax1.scatter(input_time, input_points, 
                       color='#e67e22', s=30, alpha=0.6, label='Data Points')
            ax1.set_title(f'{site} Time Series Components', pad=15)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Value')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Bottom subplot for growth rate
            ax2 = fig.add_subplot(gs[1])
            ax2.plot(input_time, growth_rate, 
                    color='#3498db', linewidth=2, label='Growth Rate')
            ax2.set_title(f'{site} Growth Rate', pad=15)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Rate')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            figures_dict[site] = (fig, (ax1, ax2))
        
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            if plot_type == 'smooth':
                ax.plot(estimation_time, smooth_value, 
                       color='#2ecc71', linewidth=2, label='Smoothed Time Series')
                ax.scatter(input_time, input_points, 
                          color='#e67e22', s=30, alpha=0.6, label='Data Points')
                ax.set_title(f'{site} Smoothed Time Series')
            
            elif plot_type == 'trend':
                ax.plot(estimation_time, trend_smooth_value, 
                       color='#e74c3c', linewidth=2, label='Trend')
                ax.scatter(input_time, input_points, 
                          color='#e67e22', s=30, alpha=0.6, label='Data Points')
                ax.set_title(f'{site} Trend Component')
            
            elif plot_type == 'growth':
                ax.plot(input_time, growth_rate, 
                       color='#3498db', linewidth=2, label='Growth Rate')
                ax.set_title(f'{site} Growth Rate')
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            figures_dict[site] = (fig, ax)
    
    return figures_dict

def save_plots(
    fig: Figure,
    filename: str,
    formats: list = ['png', 'pdf'],
    dpi: int = 300
) -> None:
    """
    Saves the figure in multiple formats with high quality settings.
    
    Args:
        fig: Matplotlib figure object
        filename: Base filename without extension
        formats: List of file formats to save
        dpi: Resolution for raster formats
    """
    for fmt in formats:
        fig.savefig(f'{filename}.{fmt}', 
                   dpi=dpi, 
                   bbox_inches='tight', 
                   pad_inches=0.1)

# %% Residual Statistics
def print_results(ccgcrv_object):
    print(ccgcrv_object.stats())
    print(ccgcrv_object.getAmplitudes())


def check_consecutive_co_sd(df, threshold_ppb=2.0, consecutive_hours=6):
    """
    Check if CO SD minutes are below threshold for specified consecutive hours.
    
    Args:
        df: pandas DataFrame containing CO data
        threshold_ppb: maximum allowed SD in ppb (default 2.0)
        consecutive_hours: number of consecutive hours required (default 6)
    
    Returns:
        pandas Series: Boolean mask where True indicates points that are part of valid sequences
    """
    # Create boolean mask for SD threshold condition
    sd_condition = df['co_SD_minutes_SCI'] < threshold_ppb
    
    # Convert to array of 0s and 1s
    condition_array = sd_condition.astype(int).values
    
    # Create array to store results
    result = np.zeros_like(condition_array)
    
    # Check for consecutive hours
    for i in range(len(condition_array) - consecutive_hours + 1):
        # If we find consecutive True values of required length
        if sum(condition_array[i:i + consecutive_hours]) == consecutive_hours:
            # Mark all points in this sequence as True
            result[i:i + consecutive_hours] = 1
    
    # Return as boolean mask
    return pd.Series(result == 1, index=df.index)

# Example usage:
# mask = check_consecutive_co_sd(df)
# filtered_df = df[mask]  # Get only the rows that meet the criteria

# To also print the timestamps of valid periods:
def print_valid_periods(df, mask):
    """Print the start and end times of valid periods."""
    valid_periods = df[mask]['datetime_UTC']
    if len(valid_periods) > 0:
        print("Valid periods where CO SD < 2 ppb for 6 consecutive hours:")
        current_start = valid_periods.iloc[0]
        prev_time = valid_periods.iloc[0]
        
        for time in valid_periods.iloc[1:]:
            if (time - prev_time) > pd.Timedelta(hours=1):
                print(f"From {current_start} to {prev_time}")
                current_start = time
            prev_time = time
        
        # Print the last period
        print(f"From {current_start} to {prev_time}")

#%%
def return_background_smoothed(gas: str = 'co2', sites: list = None):
    """
    Returns filtered and unfiltered statistics for smoothed curve fitting of gas observations.
    The function computes statistics iteratively until all observations are within 2SD of the smooth curve
    with a maximum of 7 iterations. It supports CO2, CH4, and CO gases.
    
    This can be used to compute this for computing background or smoothing time-series data

    Args:
        gas (str, optional): Gas type to compute data for. Options: 'co2', 'ch4', 'co'. Defaults to 'co2'.
        sites (list, optional): List of site codes to process. Defaults to ['VIC', 'SCI'].

    Returns:
        tuple: 
            - predicted_filtered (dict): Dictionary of predicted smooth statistics for filtered data by site.
            - original_filtered (dict): Dictionary of original statistics for filtered data by site.
            - predicted_unfiltered (dict): Dictionary of predicted smooth statistics for unfiltered data by site.
            - original_unfiltered (dict): Dictionary of original statistics for unfiltered data by site.
    """
    # Set default sites if none provided
    
    if gas == 'co':
        sites = ['SCI']
    elif gas == 'co2':
        sites = ['VIC', 'SCI']
    elif gas == 'ch4':
        sites = ['VIC', 'SCI']
    else:
        # Handle invalid gas input
        raise ValueError(f"Invalid gas input: {gas}. Valid options are 'co', 'co2', and 'ch4'.") 
  
    
    filtered_status = [True, False]  # Filtered and unfiltered data
    


    # Initialize result containers as dictionaries
    predicted_filtered = {}
    original_filtered = {}
    predicted_unfiltered = {}
    original_unfiltered = {}
    
    gas_upper = gas.upper()
    prm.path_for_files = prm.DATA_DIR + '/' + 'background_files' + '/'
    print(prm.DATA_DIR)
    print(prm.path_for_files)
    # Loop over sites and gas observations
    
    for site in sites:
        site_data_filtered_filled = compute_background_filters(
            prm.path_for_files,
            gas,
            prm.gas_dict[gas],
            prm.begin_year,
            prm.end_year,
            prm.time_frequency,
            site,
            prm.site_dict[site]  # inlet height
        )
        
        site_data_snapshot = copy.deepcopy(site_data_filtered_filled)  # Snapshot for reset
        
        for filtered in filtered_status:
            # Handle filtered data
            if filtered:
                filtered_data, unfiltered_data = background_gas_type(
                    site_data_filtered_filled,
                    gas,
                    prm.gas_num_hours[site],
                    filtered
                )
                return_values = background_return(
                    site, gas, prm.gas_dict[gas], filtered_data
                )
                original_stat = return_values[0]
                predicted_stat = return_values[1]
                
                # Store results in dictionaries with site as key
                original_filtered[site] = original_stat
                predicted_filtered[site] = predicted_stat
                
            # Handle unfiltered data
            else:
                site_data_filtered_filled = add_decimal_dates(site_data_filtered_filled)
                filtered_data = site_data_filtered_filled.dropna()
                return_values = background_return(
                    site, gas, prm.gas_dict[gas], filtered_data
                )
                original_stat = return_values[0]
                predicted_stat = return_values[1]
                
                # Store results in dictionaries with site as key
                original_unfiltered[site] = original_stat
                predicted_unfiltered[site] = predicted_stat
                
                # Reset site_data for the next iteration
                site_data_filtered_filled = copy.deepcopy(site_data_snapshot)

    return predicted_filtered, original_filtered, predicted_unfiltered, original_unfiltered
#%%
def create_time_series_plots(
    predicted_stats: dict,
    original_stats: dict,
    figsize: Tuple[int, int] = (12, 8),
    plot_type: str = 'all',
    style: str = 'seaborn',
    dpi: int = 100
) -> Dict[str, Tuple[Figure, Union[Axes, Tuple[Axes, Axes]]]]:
    """
    Creates publication-quality time series plots with multiple components for multiple sites.
    
    Args:
        predicted_stats: Dictionary of DataFrames containing predicted statistics, keyed by site
        original_stats: Dictionary of DataFrames containing original data, keyed by site
        figsize: Tuple of figure dimensions (width, height)
        plot_type: Type of plot to generate ('all', 'smooth', 'growth', 'trend')
        style: Matplotlib style to use
        dpi: Dots per inch for the figure
        
        # Get the data this is how to call the function
        predicted_filtered, original_filtered, predicted_unfiltered, original_unfiltered = return_figure_data()
        
        # Create plots for filtered data
        filtered_plots = create_time_series_plots(
            predicted_filtered,
            original_filtered,
            plot_type='all'
        )
        
        # Access specific site's figure and axes
        vic_fig, vic_axes = filtered_plots['VIC']
        sci_fig, sci_axes = filtered_plots['SCI']
        
        # Display or save figures
        plt.show()
    
    Returns:
        Dictionary of (Figure, Axes) tuples keyed by site
    """
    # List available styles for reference
    available_styles = plt.style.available
    
    # Validate style
    if style not in available_styles:
        print(f"Warning: Style '{style}' not found. Using 'default' style.")
        print(f"Available styles: {available_styles}")
        style = 'default'
    
    # Required columns validation
    required_predicted_cols = ['decimal_time', 'smooth_value', 'trend']
    required_original_cols = ['decimal_time', 'input_dependent_variable', 'smooth_detrended_cycle']
    
    # Set style
    with plt.style.context(style):  # Use context manager for style
        # Store figures for each site
        figures_dict = {}
        
        # Create plots for each site
        for site in predicted_stats.keys():
            predicted_stat = predicted_stats[site]
            original_stat = original_stats[site]
            
            # Validate columns for each site
            if not all(col in predicted_stat.columns for col in required_predicted_cols):
                raise ValueError(f"predicted_stat for {site} missing required columns: {required_predicted_cols}")
            if not all(col in original_stat.columns for col in required_original_cols):
                raise ValueError(f"original_stat for {site} missing required columns: {required_original_cols}")

            # Extract data
            estimation_time = predicted_stat['decimal_time'].to_numpy()
            smooth_value = predicted_stat['smooth_value'].to_numpy()
            trend_smooth_value = predicted_stat['trend'].to_numpy()
            input_time = original_stat['decimal_time'].to_numpy()
            input_points = original_stat['input_dependent_variable'].to_numpy()
            growth_rate = original_stat['smooth_detrended_cycle'].to_numpy()

            if plot_type == 'all':
                fig = plt.figure(figsize=figsize, dpi=dpi)
                gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)
                
                # Top subplot for time series and trend
                ax1 = fig.add_subplot(gs[0])
                ax1.plot(estimation_time, smooth_value, 
                         color='#2ecc71', linewidth=2, label='Smoothed Time Series')
                ax1.plot(estimation_time, trend_smooth_value, 
                         color='#e74c3c', linewidth=2, label='Trend')
                ax1.scatter(input_time, input_points, 
                           color='#e67e22', s=30, alpha=0.6, label='Data Points')
                ax1.set_title(f'{site} Time Series Components', pad=15)
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Value')
                ax1.grid(True, alpha=0.3)
                ax1.legend()

                # Bottom subplot for growth rate
                ax2 = fig.add_subplot(gs[1])
                ax2.plot(input_time, growth_rate, 
                        color='#3498db', linewidth=2, label='Growth Rate')
                ax2.set_title(f'{site} Growth Rate', pad=15)
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Rate')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                figures_dict[site] = (fig, (ax1, ax2))
            
            else:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                
                if plot_type == 'smooth':
                    ax.plot(estimation_time, smooth_value, 
                           color='#2ecc71', linewidth=2, label='Smoothed Time Series')
                    ax.scatter(input_time, input_points, 
                              color='#e67e22', s=30, alpha=0.6, label='Data Points')
                    ax.set_title(f'{site} Smoothed Time Series')
                
                elif plot_type == 'trend':
                    ax.plot(estimation_time, trend_smooth_value, 
                           color='#e74c3c', linewidth=2, label='Trend')
                    ax.scatter(input_time, input_points, 
                              color='#e67e22', s=30, alpha=0.6, label='Data Points')
                    ax.set_title(f'{site} Trend Component')
                
                elif plot_type == 'growth':
                    ax.plot(input_time, growth_rate, 
                           color='#3498db', linewidth=2, label='Growth Rate')
                    ax.set_title(f'{site} Growth Rate')
                
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                figures_dict[site] = (fig, ax)
    
    return figures_dict
#%% one time_series

def plot_time_series(predicted_stat, predicted_key, original_stat, original_key, value_column):
    """
    Plots the time series for the given predicted and original statistics.
    Parameters:
    - predicted_stat (dict): Dictionary containing DataFrames for predicted data
    - predicted_key (str): Key to access the DataFrame in predicted_stat
    - original_stat (dict): Dictionary containing DataFrames for original data
    - original_key (str): Key to access the DataFrame in original_stat
    - value_column (str): Column name to use for y-axis values
    """
    # Extract the DataFrames from the dictionaries
    predicted_df = predicted_stat[predicted_key]
    original_df = original_stat[original_key]
    
    # Plot predicted and original stats against datetime_UTC
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_df['datetime_UTC'], predicted_df[value_column], label=f"{predicted_key}")
    plt.plot(original_df['datetime_UTC'], original_df[value_column], label=f"{original_key}")
    
    # Add labels, legend, and title
    plt.xlabel('Datetime UTC')
    plt.ylabel(value_column)
    plt.title(f"Time Series Plot for {predicted_key} and {original_key}")
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()
    

# %%
def makefigure(predicted_filtered:pandas.DataFrame, original_filtered:pandas.DataFrame,
               predicted_unfiltered:pandas.DataFrame, original_unfiltered:pandas.DataFrame):

    trows = len(original_filtered)
    tcols = 3
    fig = make_subplots(rows=trows, cols=tcols)
    for j in range(tcols):
        if j == 0:
            for k in range(len(predicted_unfiltered)):
                # figure 1, row 1
                fig.add_trace(
                    go.Scatter(x=original_filtered[k].datetime_UTC, y=original_filtered[k].input_dependent_variable,
                               mode='markers'), row=k + 1, col=j + 1)
                fig.add_trace(go.Scatter(x=predicted_filtered[k].datetime_UTC, y=predicted_filtered[k].smooth_value,
                                         mode='lines'), row=k + 1, col=j + 1)
        if j == 1:
            for k in range(len(predicted_unfiltered)):
                # figure 1, row 1
                fig.add_trace(
                    go.Scatter(x=original_unfiltered[k].datetime_UTC, y=original_unfiltered[k].input_dependent_variable,
                               mode='markers'), row=k + 1, col=j + 1)
                fig.add_trace(go.Scatter(x=predicted_unfiltered[k].datetime_UTC, y=predicted_unfiltered[k].smooth_value,
                                         mode='lines'), row=k + 1, col=j + 1)
        if j == 2:
            for k in range(len(predicted_unfiltered)):
                # figure 1, row 1
                fig.add_trace(
                    go.Scatter(x=predicted_filtered[k].datetime_UTC, y=predicted_filtered[k].growthrate_value,
                               mode='lines'), row=k + 1, col=j + 1)
                fig.add_trace(
                    go.Scatter(x=predicted_unfiltered[k].datetime_UTC, y=predicted_unfiltered[k].growthrate_value,
                               mode='lines'), row=k + 1, col=j + 1)

    # if j == 1:
    #     for k in range(len(site)):
    #         for i in range(trows):
    #             # figure 1, row 1
    #             fig.add_trace(
    #                 go.Scatter(x=original_unfiltered[k].datetime_UTC,
    #                            y=original_unfiltered[k].input_dependent_variable,
    #                            mode='markers'), row=i+1, col=j+1)
    #             fig.add_trace(
    #                 go.Scatter(x=predicted_unfiltered[k].datetime_UTC, y=predicted_unfiltered[k].smooth_value,
    #                            mode='lines'), row=i+1, col=j+1)
    #
    # if j == 2:
    #     for k in range(len(site)):
    #         for i in range(trows):
    #             # figure 1, row 1
    #             fig.add_trace(go.Scatter(x=data.datetime_UTC, y=data.growthrate_value,
    #                                      mode='lines'), row=i+1, col=j+1)

    fig.show()

# A = back.compute_background_filters(prm.path_for_files,'co2',prm.meas_units,prm.begin_year,prm.end_year,prm.time_frequency,'VIC','100m')
# B,C=back.background_gas_type(A,'co2',6,True)
# ccg_data=back.background_return('SCI','co2','ppm',B)
# orig_stat, pred_stat, monthly_pred, mm = ccg_data
### This would provided all the data for background calculation
# predicted_filtered, original_filtered, predicted_unfiltered, original_unfiltered = back.return_figure_data('co2')
# predicted_filtered, original_filtered, predicted_unfiltered, original_unfiltered = return_figure_data('co2')
# gas = 'co2'
# site=['VIC']
# i=0
# data_site = cfunc.collect_allsite_data(prm.path_for_files, gas,
#                                                 'ppm', prm.begin_year, prm.end_year,
#                                                 prm.time_frequency, site, ['100m'])


# site_data_filtered_filled = compute_background_filters(prm.path_for_files,
#                                                         gas,  # gas type
#                                                         prm.gas_dict[gas],  # measn_units
#                                                         prm.begin_year,
#                                                         prm.end_year,
#                                                         prm.time_frequency,
#                                                         site[i],
#                                                         prm.site_dict[site[i]],  # inlet height
#                                                         )

# site_data_filtered_filled_new = copy.deepcopy(site_data_filtered_filled)
# filtered_data, unfiltered_data = background_gas_type(site_data_filtered_filled, gas,prm.gas_num_hours[site[i]],True)
# original_stat, predicted_stat, monthly_means, filtered_results = background_return(site[i],gas,prm.gas_dict[gas],filtered_data)