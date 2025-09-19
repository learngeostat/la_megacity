import numpy as np
import scipy as sc
import pandas as pd
from astropy.stats import biweight_location, biweight_midvariance, biweight_scale, median_absolute_deviation
from statsmodels.stats.stattools import medcouple
from typing import Tuple


def jaccard_distance(a, b, normalize_type):
    """
    Compute the Jaccard distance between two arrays.

    The Jaccard distance measures dissimilarity between two datasets.

    Args:
        a (numpy.ndarray): A one-dimensional array.
        b (numpy.ndarray): A one-dimensional array with the same length as array `a`.
        normalize_type (str): A string indicating whether to normalize the results ('normalized').

    Returns:
        tuple: Jaccard correlation and other quantities (a_intersection_b, a_minus_b, b_minus_a, a_union_b).

    Raises:
        ValueError: If the input arrays do not have the same length.
        TypeError: If the inputs are not numpy arrays.

    References:
        Jaccard, P. (1901). Etude comparative de la distribution florale dans une portion des Alpes et du Jura.
        Bulletin de la Société Vaudoise des Sciences Naturelles, 37, 547-579.
    """
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if len(a) != len(b):
        raise ValueError("Input arrays must have the same length.")

    ab = np.column_stack((a, b))
    ab = ab[~np.isnan(ab).any(axis=1), :]
    a_intersection_b = np.sum(np.amin(ab, axis=1))
    a_plus_b = np.sum(ab)
    a_union_b = a_plus_b - a_intersection_b
    a_minus_b = a_union_b - np.nansum(ab[:, 1])
    b_minus_a = a_union_b - np.nansum(ab[:, 0])
    if normalize_type == 'normalized':
        a_minus_b = a_minus_b / a_union_b
        b_minus_a = b_minus_a / a_union_b
        a_intersection_b = a_intersection_b / a_union_b
    return a_intersection_b, a_minus_b, b_minus_a, a_union_b


def pearson_kappa(data):
    """
    Calculate the Pearson Kappa criterion for distribution fit.

    Args:
        data (numpy.ndarray): A one-dimensional array of numerical data.

    Returns:
        float: Kappa fit value.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input array is empty or contains non-numeric values.

    References:
        Elderton, W. P., & Johnson, N. L. (1969). Systems of Frequency Curves.
        Cambridge University Press.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if len(data) == 0 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input array must be non-empty and contain only numeric values.")

    m2 = sc.stats.moment(data, 2)
    m3 = sc.stats.moment(data, 3)
    m4 = sc.stats.moment(data, 4)
    b1 = (m3 ** 2) / (m2 ** 3)
    b2 = m4 / (m2 ** 2)
    kappa = (b1 * (b2 + 3) ** 2) / (4 * (4 * b2 - 3 * b1) * (2 * b2 - 3 * b1 - 6))
    return kappa


def gini(x, w=None):
    """
    Calculate the Gini coefficient for a given dataset.

    Args:
        x (numpy.ndarray): A one-dimensional array with positive values.
        w (numpy.ndarray, optional): A one-dimensional array with weights.

    Returns:
        float: Gini coefficient.

    Raises:
        TypeError: If the input is not a numpy array or contains non-numeric values.
        ValueError: If the input array is empty.

    References:
        Gini, C. (1912). Variability and Mutability, Contribution to the Study of Statistical Distributions and Relations.
        C. Cuppini, Bologna.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if len(x) == 0 or not np.issubdtype(x.dtype, np.number):
        raise ValueError("Input array must be non-empty and contain only numeric values.")

    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def sn_stat(data):
    """
    Calculate the S_n statistic for robust scale estimation.

    Args:
        data (numpy.ndarray): A one-dimensional array of numerical data.

    Returns:
        float: S_n statistic.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input array is empty or contains non-numeric values.

    References:
        Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the Median Absolute Deviation.
        Journal of the American Statistical Association, 88(424), 1273-1283.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if len(data) == 0 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input array must be non-empty and contain only numeric values.")

    sn_stat_data = np.nan * np.ones((len(data), len(data)))
    for i in range(len(data)):
        sn_stat_data[:, i] = abs(data[i] - data)
        sn_stat_data[i, i] = np.nan
    stat = np.nanmedian(np.nanmedian(sn_stat_data, axis=0)) * 1.1926
    return stat


def qn_stat(data):
    """
    Calculate the Q_n statistic for robust scale estimation.

    Args:
        data (numpy.ndarray): A one-dimensional array of numerical data.

    Returns:
        float: Q_n statistic.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input array is empty or contains non-numeric values.

    References:
        Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the Median Absolute Deviation.
        Journal of the American Statistical Association, 88(424), 1273-1283.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if len(data) == 0 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input array must be non-empty and contain only numeric values.")

    data_sort = np.sort(data)
    qn_stat_data = np.nan * np.ones((len(data), len(data)))
    for i in range(len(data)):
        qn_stat_data[i + 1:len(data), i] = abs(data_sort[i] - data_sort[i + 1:len(data)])
        qn_stat_data[i, i] = np.nan
    stat = np.nanmedian(np.nanmedian(qn_stat_data, axis=0))
    return stat


def location_stat(data):
    """
    Calculate various location statistics.

    Args:
        data (numpy.ndarray): A one-dimensional array of numerical data.

    Returns:
        pandas.DataFrame: DataFrame containing location statistics.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input array is empty or contains non-numeric values.

    References:
        Elderton, W. P., & Johnson, N. L. (1969). Systems of Frequency Curves.
        Cambridge University Press.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if len(data) == 0 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input array must be non-empty and contain only numeric values.")

    mean = np.nanmean(data)
    trimmed_mean = sc.stats.trim_mean(data, 0.1)
    median = np.nanmedian(data)
    geometric_mean = sc.stats.gmean(data)
    biweight_loc = biweight_location(data)
    lower_quartile = np.percentile(data, 25)
    upper_quartile = np.percentile(data, 75)
    interquartile_mean = np.nanmean(data[(data >= lower_quartile) & (data <= upper_quartile)])
    loc_data = [mean, trimmed_mean, median, geometric_mean, biweight_loc, interquartile_mean]
    loc_names = ['mean', 'trimmed_mean', 'median', 'geometric_mean', 'biweight_location', 'interquartile_mean']
    statistics = pd.DataFrame({'central_tendency_name': loc_names, 'estimate': loc_data})

    return statistics


def dispersion_stat(data):
    """
    Calculate various dispersion statistics.
    Args:
        data (numpy.ndarray): A one-dimensional array of numerical data.
    Returns:
        pandas.DataFrame: DataFrame containing dispersion statistics.
    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input array is empty or contains non-numeric values.
    References:
        Elderton, W. P., & Johnson, N. L. (1969). Systems of Frequency Curves.
        Cambridge University Press.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if len(data) == 0 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input array must be non-empty and contain only numeric values.")

    # Calculate statistics
    std = np.std(data)
    iqr = sc.stats.iqr(data)
    trimmed_var = sc.stats.mstats.trimmed_var(data, limits=(0.1, 0.1))
    mean_absolute_deviation = np.mean(np.absolute(data - np.mean(data)))
    mad = median_absolute_deviation(data)  # Renamed variable to avoid conflict
    mean_log_deviation = np.log(np.mean(np.absolute(data))) - np.mean(np.log(np.absolute(data)))
    biweight_midvar = biweight_midvariance(data)
    biweight_scale_val = biweight_scale(data)
    gini_coefficient = gini(abs(data))
    std_log_data = np.sqrt((sum((np.log(data) - np.mean(np.log(np.absolute(data)))) ** 2)) / len(data))

    # Compile results
    dispersion_data = [std, iqr, trimmed_var, mean_absolute_deviation, mad, mean_log_deviation,
                       biweight_midvar, biweight_scale_val, gini_coefficient, std_log_data]
    dispersion_names = ['std', 'iqr', 'trimmed_var', 'mean_absolute_deviation', 'median_absolute_deviation',
                        'mean_log_deviation', 'biweight_midvariance', 'biweight_scale', 'gini_coefficient',
                        'std_log_data']
    
    # Create DataFrame
    statistics = pd.DataFrame({'dispersion_stat_name': dispersion_names, 'estimate': dispersion_data})
    return statistics


def order_stat(data):
    """
    Calculate various order statistics.

    Args:
        data (numpy.ndarray): A one-dimensional array of numerical data.

    Returns:
        pandas.DataFrame: DataFrame containing order statistics.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input array is empty or contains non-numeric values.

    References:
        Elderton, W. P., & Johnson, N. L. (1969). Systems of Frequency Curves.
        Cambridge University Press.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if len(data) == 0 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input array must be non-empty and contain only numeric values.")

    minimum = np.min(data)
    first_quartile = np.percentile(data, 25)
    median = np.percentile(data, 50)
    third_quartile = np.percentile(data, 75)
    maximum = np.max(data)
    mid_range = (maximum + minimum) / 2
    data_range = maximum - minimum
    order_stat = [minimum, first_quartile, median, third_quartile, maximum, mid_range, data_range]
    order_names = ['minimum', 'first_quartile', 'median', 'third_quartile', 'maximum', 'mid_range', 'data_range']
    statistics = pd.DataFrame({'order_stat_name': order_names, 'estimate': order_stat})

    return statistics


def shape_stat(data):
    """
    Calculate various shape statistics.

    Args:
        data (numpy.ndarray): A one-dimensional array of numerical data.

    Returns:
        pandas.DataFrame: DataFrame containing shape statistics.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input array is empty or contains non-numeric values.

    References:
        Elderton, W. P., & Johnson, N. L. (1969). Systems of Frequency Curves.
        Cambridge University Press.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if len(data) == 0 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input array must be non-empty and contain only numeric values.")

    skew = sc.stats.skew(data)
    kurtosis = sc.stats.kurtosis(data)
    mean = np.mean(data)
    standard_deviation = np.std(data)
    second_quartile = np.percentile(data, 75)
    median = np.percentile(data, 50)
    first_quartile = np.percentile(data, 25)
    first_decile = np.percentile(data, 10)
    last_decile = np.percentile(data, 90)
    bowley_skewness = ((second_quartile + first_quartile) - (2 * median)) / (second_quartile - first_quartile)
    kelly_skewness = ((last_decile + first_decile) - (2 * median)) / (last_decile - first_decile)
    pearson_second_skew = (3 * mean - median) / standard_deviation
    entropy = sc.stats.entropy(data)
    median_couple = np.round(medcouple(data), 6)
    shape_stat = [skew, kurtosis, bowley_skewness, kelly_skewness, pearson_second_skew, median_couple, entropy]
    shape_stat_names = ['skewness', 'kurtosis', 'bowley_skewness', 'kelly_skewness', 'pearson_second_skew', 'medcouple',
                        'entropy']
    statistics = pd.DataFrame({'summary_stat_name': shape_stat_names, 'estimate': shape_stat})
    return statistics


def summary_stat(data):
    """
    Calculate various summary statistics.

    Args:
        data (numpy.ndarray): A one-dimensional array of numerical data.

    Returns:
        pandas.DataFrame: DataFrame containing summary statistics.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input array is empty or contains non-numeric values.

    References:
        Elderton, W. P., & Johnson, N. L. (1969). Systems of Frequency Curves.
        Cambridge University Press.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if len(data) == 0 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input array must be non-empty and contain only numeric values.")

    count = len(data)
    minimum = np.min(data)
    maximum = np.max(data)
    mean = np.nanmean(data)
    median = np.percentile(data, 50)
    std = np.std(data)
    inter_quartile_range = sc.stats.iqr(data)
    data_range = maximum - minimum
    data_sum = np.sum(data)
    summary_stat = [count, minimum, maximum, mean, median, std, inter_quartile_range, data_range, data_sum]
    summary_names = ['count', 'minimum', 'maximum', 'mean', 'median', 'std', 'inter_quartile_range', 'data_range',
                     'data_sum']
    statistics = pd.DataFrame({'summary_stat_name': summary_names, 'estimate': summary_stat})
    return statistics

def essential_stats(data):
    """
    Calculate essential statistics that are reliable even for small datasets.
    Includes basic distributional characteristics and robust measures.
    
    Args:
        data (numpy.ndarray): A one-dimensional array of numerical data.
    
    Returns:
        pandas.DataFrame: DataFrame containing essential statistics.
    
    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input array is empty or contains non-numeric values.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if len(data) == 0 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input array must be non-empty and contain only numeric values.")

    # Calculate quantiles once and reuse
    quantiles = np.percentile(data, [0, 10, 25, 50, 75, 90, 100])
    minimum, first_decile, first_quartile, median, third_quartile, last_decile, maximum = \
        [np.round(x, 3) for x in quantiles]
    
    # Basic statistics
    count = int(len(data))
    mean = np.round(np.nanmean(data), 3)
    
    # Spread measures
    std = np.round(np.std(data), 3)
    iqr = np.round(third_quartile - first_quartile, 3)
    mad = np.round(median_absolute_deviation(data), 3)
    
    # Shape measures
    skewness = np.round(sc.stats.skew(data), 3)
    kurtosis = np.round(sc.stats.kurtosis(data, fisher=False), 3)
    
    # Compile statistics
    stat_names = [
        'count',
        'minimum', 'maximum',
        'first_decile', 'last_decile',
        'first_quartile', 'third_quartile',
        'mean', 'median',
        'std', 'iqr', 'mad',
        'skewness', 'kurtosis'
    ]
    
    stat_values = [
        count,
        minimum, maximum,
        first_decile, last_decile,
        first_quartile, third_quartile,
        mean, median,
        std, iqr, mad,
        skewness, kurtosis
    ]
    
    # Create DataFrame
    statistics = pd.DataFrame({'STAT_NAME': stat_names, 'ESTIMATE': stat_values})
    return statistics


def mass_stat(data):
    """
    Calculate mass statistics at various percentiles.

    Args:
        data (numpy.ndarray): A one-dimensional array of numerical data.

    Returns:
        pandas.DataFrame: DataFrame containing mass statistics at various percentiles.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input array is empty or contains non-numeric values.

    References:
        Elderton, W. P., & Johnson, N. L. (1969). Systems of Frequency Curves.
        Cambridge University Press.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if len(data) == 0 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input array must be non-empty and contain only numeric values.")

    data_sum = np.sum(data)
    mass_stat = []
    quartile_stat = []
    mass_names = ['5', '10', '20', '25', '30', '40', '50', '60', '70', '75', '80', '90', '95', '99', '100']
    mass_val = [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99, 100]
    for i in mass_val:
        cutof = np.percentile(data, i)
        quartile_stat.append(cutof)
        estimate = np.round(np.sum(data[data <= cutof]) / data_sum, 3)
        mass_stat.append(estimate)

    statistics = pd.DataFrame(
        {'summary_quantile_name': mass_names, 'quantile_val': quartile_stat, 'estimate': mass_stat})
    return statistics


def all_stat(data):
    """
    Calculate a comprehensive set of statistics including location, dispersion, shape, and order statistics.

    Args:
        data (numpy.ndarray): A one-dimensional array of numerical data.

    Returns:
        pandas.DataFrame: DataFrame containing all computed statistics.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input array is empty or contains non-numeric values.

    References:
        Elderton, W. P., & Johnson, N. L. (1969). Systems of Frequency Curves.
        Cambridge University Press.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if len(data) == 0 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input array must be non-empty and contain only numeric values.")

    # Basic statistics
    count = int(len(data))
    minimum = np.round(np.min(data), 3)
    maximum = np.round(np.max(data), 3)
    data_range = maximum - minimum
    data_sum = np.sum(data)
    mean = np.round(np.nanmean(data), 3)
    median = np.round(np.percentile(data, 50), 3)
    
    # Location measures
    trimmed_mean = np.round(sc.stats.trim_mean(data, 0.1), 3)
    geometric_mean = np.round(sc.stats.gmean(data), 3)
    biweit_location = np.round(biweight_location(data), 3)
    
    # Dispersion measures
    std = np.round(np.std(data), 3)
    inter_quartile_range = np.round(sc.stats.iqr(data), 3)
    trimmed_var = np.round(sc.stats.mstats.trimmed_var(data, limits=(0.1, 0.1)), 3)
    mean_absolute_deviation = np.round(np.mean(np.absolute(data - np.mean(data))), 3)
    mad = np.round(median_absolute_deviation(data), 3)  # Renamed to avoid conflict
    mean_log_deviation = np.round(np.log(np.mean(np.absolute(data))) - np.mean(np.log(np.absolute(data))), 3)
    biweight_midvar = np.round(biweight_midvariance(data), 3)  # Renamed to avoid conflict
    biweight_scale_val = np.round(biweight_scale(data), 3)
    gini_coefficient = np.round(gini(abs(data)), 3)
    std_log_data = np.round(np.sqrt((sum((np.log(data) - np.mean(np.log(np.absolute(data)))) ** 2)) / len(data)), 3)
    
    # Quartiles and deciles
    first_quartile = np.round(np.percentile(data, 25), 3)
    second_quartile = np.round(np.percentile(data, 75), 3)
    first_decile = np.round(np.percentile(data, 10), 3)
    last_decile = np.round(np.percentile(data, 90), 3)
    mid_range = np.round((maximum + minimum) / 2, 3)
    
    # Shape measures
    skew = np.round(sc.stats.skew(data), 2)
    bowley_skewness = np.round(((second_quartile + first_quartile) - (2 * median)) / (second_quartile - first_quartile), 3)
    kelly_skewness = np.round(((last_decile + first_decile) - (2 * median)) / (last_decile - first_decile), 3)
    pearson_second_skew = np.round((3 * mean - median) / std, 3)
    entropy = np.round(sc.stats.entropy(data), 3)
    median_couple = np.round(medcouple(data), 3)
    kurtosis = np.round(sc.stats.kurtosis(data, fisher=False), 3)
    pears_kappa = np.round(pearson_kappa(data), 3)

    # Compile all statistics
    stat_names = ['count', 'minimum', 'maximum', 'data_range', 'data_sum', 'mean', 'median', 'trimmed_mean',
                  'geometric_mean', 'biweight_location', 'std', 'inter_quartile_range', 'trimmed_var', 
                  'mean_absolute_deviation', 'median_absolute_deviation', 'mean_log_deviation', 
                  'biweight_midvariance', 'biweight_scale', 'gini_coefficient', 'std_log_data', 
                  'first_quartile', 'second_quartile', 'first_decile', 'last_decile', 'mid_range', 
                  'skewness', 'bowley_skewness', 'kelly_skewness', 'pearson_second_skew', 'medcouple',
                  'entropy', 'kurtosis', 'pearson_kappa']

    all_stat_values = [count, minimum, maximum, data_range, data_sum, mean, median, trimmed_mean, geometric_mean,
                       biweit_location, std, inter_quartile_range, trimmed_var, mean_absolute_deviation, mad,
                       mean_log_deviation, biweight_midvar, biweight_scale_val, gini_coefficient, std_log_data,
                       first_quartile, second_quartile, first_decile, last_decile, mid_range, skew, 
                       bowley_skewness, kelly_skewness, pearson_second_skew, median_couple, entropy, 
                       kurtosis, pears_kappa]

    # Create and return DataFrame
    statistics = pd.DataFrame({'STAT_NAME': stat_names, 'ESTIMATE': all_stat_values})
    print(statistics)
    return statistics


def all_percentiles_discrete(data):
    """
    Calculate percentiles from 0 to 100 for a given dataset.

    Args:
        data (numpy.ndarray): A one-dimensional array of numerical data.

    Returns:
        numpy.ndarray: Array containing percentiles from 0 to 100.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input array is empty or contains non-numeric values.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if len(data) == 0 or not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input array must be non-empty and contain only numeric values.")

    percentile_data = np.nan * np.ones(101)
    for i in range(0, 101, 1):
        percentile_data[i] = np.percentile(data, i)
    return percentile_data


def quantile_factor(quantile_data_a, quantile_data_b):
    """
    Calculate the quantile factor for two datasets.

    Args:
        quantile_data_a (numpy.ndarray): Array of quantile data.
        quantile_data_b (numpy.ndarray): Array of quantile data.

    Returns:
        float: Quantile factor.

    Raises:
        TypeError: If the inputs are not numpy arrays.
        ValueError: If the input arrays are empty or have different lengths.
    """
    if not isinstance(quantile_data_a, np.ndarray) or not isinstance(quantile_data_b, np.ndarray):
        raise TypeError("Inputs must be numpy arrays.")
    if len(quantile_data_a) == 0 or len(quantile_data_b) == 0:
        raise ValueError("Input arrays must be non-empty.")
    if len(quantile_data_a) != len(quantile_data_b):
        raise ValueError("Input arrays must have the same length.")

    quantile_factor = np.nanmean(quantile_data_a / quantile_data_b)
    return quantile_factor


def symmetry_plot_data(data):
    n = len(data)
    n2 = np.floor(n / 2)
    mx = np.nanmedian(data)
    sx = np.sort(data)
    x1 = mx - sx[0:int(n2)]
    sx = np.flip(sx)
    y1 = sx[0:int(n2)] - mx
    symmetry_plot_data = np.vstack((x1, y1)).T
    symmetry_data=pd.DataFrame(symmetry_plot_data, columns=['x', 'y'])
    return symmetry_data

# import plotly.graph_objects as go
# import pandas as pd
# import plotly.io as pio
# pio.renderers.default = "browser"
# data = np.array([3, 4, 6, 10, 24, 89, 45, 43, 46, 99, 100])
# p1 = all_percentiles_discrete(data)
# p2 = all_percentiles_discrete(data+1)
# p2=np.vstack((p1,p2)).T
# df = pd.DataFrame(p2, columns = ['x','y'])
# #fig=px.scatter(df,x='x',y='y')
# rsquared=0.92
# fig=go.Figure()
# fig.add_trace(go.Scatter(x=df['x'], y=df['y'], name="Quantiles", mode="markers"))
# fig.add_trace(go.Scatter(x=df['x'], y=df['x'], name="Identity Line", line_shape='linear', mode="lines"))
# fig.update_traces(marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')))
# fig.update_layout(xaxis_title="Quantiles X", yaxis_title="Quantiles Y",
#                   margin={"r": 10, "t": 50, "l": 0, "b": 0},
#                   title_x=0.5,
#                   #legend_title="Scaling Factor",
#                   font_family="Arial",
#                   font_color="sandybrown",
#                   font_size=20,
#                   template='plotly_dark',
#                   plot_bgcolor='rgba(0, 0, 0, 0)',
#                   paper_bgcolor='rgba(0, 0, 0, 0)',
#                   legend=dict(  # yanchor="top",
#                       y=0.95,
#                       x=0.05,
#                       orientation="v",
#                       font={'family': "Arial", 'size': 20, 'color': "black"},
#                       bordercolor="rgba(147,112,219,0)",
#                       borderwidth=2,
#                   ),
#                   legend_title_font_color="maroon",
#                   title_font={'size': 18, 'color': "Black", 'family': "Arial"}
#                   )

# annotation_1 = {'xref': 'paper',  # we'll reference the paper which we draw plot
#                 'yref': 'paper',  # we'll reference the paper which we draw plot
#                 'x': 0.05,
#                 'y': 0.96,
#                 'font_family' : "Arial",
#                 'font_color' : "black",
#                 'font_size' : 20,
#                 'text': 'Scaling Factor = ' + str(round(rsquared, 2)),
#                 'showarrow': False,
# 'font': {'size': 20, 'color': "black"}
# }
# fig.update_layout({'annotations': [annotation_1]})

# fig.update_xaxes(showline=True, ticks='outside',
#                  ticklen=10, tickcolor="sandybrown", tickwidth=2,
#                  showgrid=True,
#                  zeroline=False,
#                  gridwidth=2,
#                  gridcolor='lightskyblue')

# fig.update_yaxes(showline=True, tickformat='.2f', ticks='outside',
#                  ticklen=10, tickcolor="sandybrown", tickwidth=2,
#                  zeroline=True, zerolinecolor='lightskyblue', zerolinewidth=2, showgrid=True, gridwidth=2,
#                  gridcolor='lightskyblue')

# fig.add_shape(
#     # Rectangle with reference to the plot
#     type="rect",
#     xref="paper",
#     yref="paper",
#     x0=0,
#     y0=0,
#     x1=1.0,
#     y1=1.0,
#     line=dict(
#         color="lightskyblue",
#         width=1,
#     )
# )

# fig.show()

# # Symmetry Plot
# import plotly.graph_objects as go
# import pandas as pd
# import plotly.io as pio
# pio.renderers.default = "browser"
# data = np.array([3, 4, 6, 10, 24, 89, 45, 43, 46, 99, 100])
# p1 = all_percentiles_discrete(data)
# p2 = all_percentiles_discrete(data+1)
# p2=np.vstack((p1,p2)).T
# df = pd.DataFrame(p2, columns = ['x','y'])
# #fig=px.scatter(df,x='x',y='y')
# df = symmetry_plot
# rsquared=3.30
# fig=go.Figure()
# fig.add_trace(go.Scatter(x=df['x'], y=df['y'], name="Assymetry", mode="markers"))
# fig.add_trace(go.Scatter(x=df['x'], y=df['x'], name="Identity Line", line_shape='linear', mode="lines"))
# fig.update_traces(marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')))
# fig.update_layout(xaxis_title="Quantiles X", yaxis_title="Quantiles Y",
#                   margin={"r": 10, "t": 50, "l": 0, "b": 0},
#                   title_x=0.5,
#                   #legend_title="Scaling Factor",
#                   font_family="Arial",
#                   font_color="sandybrown",
#                   font_size=20,
#                   template='plotly_dark',
#                   plot_bgcolor='rgba(0, 0, 0, 0)',
#                   paper_bgcolor='rgba(0, 0, 0, 0)',
#                   legend=dict(  # yanchor="top",
#                       y=0.95,
#                       x=0.05,
#                       orientation="v",
#                       font={'family': "Arial", 'size': 20, 'color': "black"},
#                       bordercolor="rgba(147,112,219,0)",
#                       borderwidth=2,
#                   ),
#                   legend_title_font_color="maroon",
#                   title_font={'size': 18, 'color': "Black", 'family': "Arial"}
#                   )

# annotation_1 = {'xref': 'paper',  # we'll reference the paper which we draw plot
#                 'yref': 'paper',  # we'll reference the paper which we draw plot
#                 'x': 0.05,
#                 'y': 0.96,
#                 'font_family' : "Arial",
#                 'font_color' : "black",
#                 'font_size' : 20,
#                 'text': 'Scaling Factor = ' + str(round(rsquared, 2)),
#                 'showarrow': False,
# 'font': {'size': 20, 'color': "black"}
# }
# fig.update_layout({'annotations': [annotation_1]})

# fig.update_xaxes(showline=True, ticks='outside',
#                  ticklen=10, tickcolor="sandybrown", tickwidth=2,
#                  showgrid=True,
#                  zeroline=False,
#                  gridwidth=2,
#                  gridcolor='lightskyblue')

# fig.update_yaxes(showline=True, tickformat='.2f', ticks='outside',
#                  ticklen=10, tickcolor="sandybrown", tickwidth=2,
#                  zeroline=True, zerolinecolor='lightskyblue', zerolinewidth=2, showgrid=True, gridwidth=2,
#                  gridcolor='lightskyblue')

# fig.add_shape(
#     # Rectangle with reference to the plot
#     type="rect",
#     xref="paper",
#     yref="paper",
#     x0=0,
#     y0=0,
#     x1=1.0,
#     y1=1.0,
#     line=dict(
#         color="lightskyblue",
#         width=1,
#     )
# )

# fig.show()