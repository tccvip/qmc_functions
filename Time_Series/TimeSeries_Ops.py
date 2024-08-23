'''
Time Series Operators

Sources: 
    https://platform.worldquantbrain.com/learn/operators/operators

Args(Params):
    y, x: pandas.Series  # or numpy.array
    d: numbers of look back days

'''

import pandas as pd
from pandas import Series
import numpy as np



def days_from_last_change(x: Series) -> int:
    '''
        Params: 
            x (pandas.Series): Time Series Data

        Return:
            int: Amount of days since last change of x
    '''
    last_value = x.iloc[-1]
    diffrent_day = x[x != last_value].index[-1]
    last_day = x.index[-1]

    return (last_day - diffrent_day).days if not isinstance(last_day) else last_day - diffrent_day


def ts_weighted_delay(x: Series, k=0.5) -> Series:
    '''
        Params:
            k (float) (0<=k<=1): weight of today's value
        
        Return:
            Series: new Series with weight on today’s value being k and yesterday’s being (1-k)
    '''

    return np.add(k* x, x.shift(1)*(1-k))


def hump(x: Series, hump=0.01):
    pass




def hump_decay(x: Series, p=0, relative=False):
    '''
        Returns:
            Series:  ignored the values that changed too little corresponding to previous ones
                if relative == False:
                    if abs(x[t] - x[t-1])> p, return x[t], else x[t-1]
                if relative == True:
                    if abs(x[t] - x[t-1])> p * abs(x[t] + x[t-1]), return x[t], else x[t-1]
    '''
    prev = x.iloc[0]
    res = x.copy()
    for i in range(1, len(x)):
        if (relative and abs(x.iloc[i] - prev) > p * abs(x.iloc[i] + prev)) or (not relative and abs(x.iloc[i] - prev) > p):
            res.iloc[i] = prev
        prev = x.iloc[i]
    return res


def inst_tvr(x: Series, d):
    pass




def jump_decay(x: Series, d, sensitivity=0.5, force=0.1) -> Series:
    '''
    jump_decay(x) = abs(x-ts_delay(x, 1)) > sensitivity * ts_stddev(x,d) ? ts_delay(x,1) + ts_delta(x, 1) * force: x
      
        Return:
            Series: If there is a huge jump in current data compare to previous one
    '''
    condition = abs(x-ts_delay(x, 1)) > sensitivity * ts_std_dev(x,d)
    temp = x
    temp[condition] = ts_delay(x, 1) + ts_delta(x, 1) * force
    return temp


def kth_element(x, d, k:str|int = '1', ignore: str = 'Nan') -> Series:
    '''
        Params:
            ignore (str): Space-separated list of scalars to ignore (default: 'NAN')
        Returns:
            float: The k-th value of x, ignoring specified scalars
    '''
    k = int(k)
    last_d_elements = x.iloc[-d:]
    for scalar in ignore.split():
        last_d_elements = last_d_elements.replace(scalar, np.nan)
    valid_elements = last_d_elements.dropna()

    return valid_elements.iloc[-k] if len(valid_elements) >= k else np.nan


def last_diff_value(x: Series, d) -> np.float64:
    '''
        Returns:
            dtype: last x value not equal to current x value from last d days
                Returns None if all x value from last d days are equal
    '''
    dd = days_from_last_change(x)
    return x.iloc[-dd] if dd<d else None


def ts_arg_max(x:Series, d) -> int:
    '''
        Returns:
            int (number of days): the relative index of the max value in the time series for the past d days.
        Example:
            d=6 and values for past 6 days are[6,2,8,5,9,4] -> return 4 (as max value is 9)
    '''
    last_d_days = x.iloc[-d:]
    max_index = last_d_days.argmax()

    return d - 1 - max_index


def ts_arg_min(x:Series, d) -> int:
    '''
        Returns:
            int (number of days): the relative index of the min value in the time series for the past d days.
        Example:
            d=6 and values for past 6 days are[6,2,8,5,9,4] -> return 1 (as min value is 1)     
    '''
    last_d_days = x.iloc[-d:]
    min_index = last_d_days.argmin()

    return d - 1 - min_index


def ts_av_diff(x:Series, d) -> Series:
    '''
        Returns:
            Series: x - tsmean(x, d) but nan vals were skipped
    '''
    return x - ts_mean(x, d, skipna=True)


def ts_backfill(x: Series,d, k=1, ignore="NAN") -> Series:
    '''
        Params:
            d (int) : lookback days
        Returns:
            replacing the NAN or 0 values by a meaningful value (i.e., a first non-NaN value)
    '''
    pass
    



def ts_count_nans(x:Series, d) -> np.int64:
    '''
        Returns:
             the number of NaN values in x for the past d days
    '''
    last_d_elements = x.iloc[-d:]
    return last_d_elements.isna().sum()


def ts_decay_exp_window(x: Series, d, factor=1.0, nan=False) -> Series:
    '''
        Params:
            factor (0<factor<1): smoothing factor 
            nan = True : means that nan will be returned in case if values of X is nan.
            Iparameter nan = False : and all values of X is nan, the operator will return 0.
        Returns:
            TS_Decay_Exp_Window (x, d, factor = f) = (x[date] + x[date - 1] * f + … + x[date – (d - 1)] * (f ^ (d – 1))) / (1 + f + … + f ^ (d - 1))
    '''
    def helper(window):
        if len(window) < d:
            return np.nan
        if all(np.isnan(window)):
            if nan: 
                return np.nan
            else:
                return 0
        else:
            return np.divide(np.sum(window * weights), w_sum) 
        
    weights = factor ** np.arange(d-1,-1,-1)
    w_sum = np.sum(weights)
    decayed_values = x.rolling(window=d, min_periods=0).apply(
        helper,
        raw=True,
    )
    
    return decayed_values


def ts_decay_linear(x: Series, d) -> Series:
    pass




def ts_delay(x: Series, d) -> Series:
    '''
        Return:
            Series: Returns x value d days ago
    '''
    return x.shift(d)

 
def ts_delta(x, d) -> Series:
    '''
        Return x - ts_delay(x, d)
    '''
    return x - ts_delay(x,d)


def ts_ir(x: Series, d) -> Series:
    '''
        Returns:
            Return information ratio ts_mean(x, d) / ts_std_dev(x, d)
    '''
    return np.divide(ts_mean(x, d), ts_std_dev(x,d))


def ts_kurtosis(x, d): pass




def ts_max(x: Series, d) -> Series:
    '''
        Returns:
            Series: max value of x for the past d days
    '''
    return x.rolling(window=d).max()


def ts_max_diff(x: Series, d) -> Series:
    '''
        Returns:
            Series:  x - ts_max(x, d)
    '''
    return x - ts_max(x, d)


def ts_mean(x: Series, d, skipna=False) -> Series:
    '''
        Params:
            skipna (bool): Whether to skip NaN values when calculating the average (default: False)
        Returns:
            Series:  average value of x for the past d days.
    '''
    if not skipna:
        rolling_mean = x.rolling(window=d).mean()
    else:
        rolling_mean = x.rolling(window=d, min_periods=1).mean()

    return rolling_mean


def ts_median(x: Series, d) -> Series:
    '''
        Returns:
            Series: median value of x for the past d days
    '''
    return x.rolling(window=d).median()


def ts_min(x: Series, d) -> Series:
    '''
        Returns:
            Series: min value of x for the past d days
    '''
    return x.rolling(window=d).min()


def ts_min_diff(x: Series, d) -> Series:
    '''
        Returns:
            Series: x - ts_min(x, d)
    '''
    return x - ts_min(x, d)


def ts_min_max_cps(x: Series, d, f = 2) -> Series:
    '''
        Returns:
            Series: (ts_min(x, d) + ts_max(x, d)) - f * x
    '''
    return (ts_min(x,d) + ts_max(x, d)) - f*x


def ts_min_max_diff(x: Series, d, f = 0.5) -> Series:
    '''
        Returns:
            Series:  x - f * (ts_min(x, d) + ts_max(x, d)
    '''
    return x - f * (ts_min(x,d) + ts_max(x,d))


def ts_moment(x, d, k=0) -> Series:
    '''
        Returns:
            Series: K-th central moment of x for the past d days
                 K-th central moment: https://egyankosh.ac.in/bitstream/123456789/20443/1/Unit-3.pdf page 57
    '''
    new = ts_av_diff(x, d) ** k
    return new.rolling(window=d).mean()


def ts_partial_corr(x, y, z, d): pass




def ts_percentage(x: Series, d, percentage=0.5) -> Series:
    '''
        Returns:
            Series: percentile value of x for the past d days
    '''
    return x.rolling(window=d).quantile(q=percentage)


def ts_poly_regression(y, x, d, k = 1): pass




def ts_product(x: Series, d) -> Series:
    '''
        Returns:
            Series:  product of x for the past d days
    '''
    return x.rolling(window=d).apply(
        lambda window: np.nanprod(window),
        raw=True
    )


def ts_rank(x, d, constant = 0): pass




def ts_regression(y, x, d, lag = 0, rettype = 0): pass




def ts_returns (x: Series, d, mode = 1) -> Series:
    '''
        Returns:
            If mode = 1, it returns (x – ts_delay(x, d )) / ts_delay(x, d)
            If mode = 2, it returns mode == 2: (x – ts_delay(x, d )) / ((x + ts_delay(x, d))/2)
    '''        
    if mode in [1,'1']:
        return  np.divide(x - ts_delay(x, d ), ts_delay(x, d))
    elif mode in [2,'2']:
        return np.divide(x - ts_delay(x,d), (x + ts_delay(x,d))/2 )
    else:
        raise ValueError("mode should be one of 1,2")


def ts_scale(x: Series, d, constant = 0) -> Series:
    '''
        Returns:
            Series:  (x – ts_min(x, d)) / (ts_max(x, d) – ts_min(x, d)) + constant
    '''
    return np.divide(x - ts_min(x, d), ts_max(x, d) - ts_min(x, d)) + constant


def ts_skewness(x,d) -> Series: pass




def ts_step():
    '''
    https://support.worldquantbrain.com/hc/en-us/community/posts/18280724186519-ts-step-using
    https://support.worldquantbrain.com/hc/en-us/community/posts/13580036296087-Use-of-operator-ts-step
    '''
    pass




def ts_sum(x: Series, d) -> Series:
    '''
        Returns:
            Series: Sum values of x for the past d days.
    '''
    return x.rolling(window=d).sum()


def ts_std_dev(x: Series, d) -> Series:
    '''
        Return:
            Series: Returns standard deviation of x for the past d days

                Standard Deviation:
                    Euclidean distance of x and a series where all values are mean value of x 
    '''

    return x.rolling(window=d).std()





import time
dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
values = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
series_1 = pd.Series([2,4,4,4,5,5,7,9,5,6], index=dates)
series_2 = pd.Series(values, index=dates)



df_1 = pd.DataFrame([2,4,4,4,5,5,7,9])
# x = pd.Series([np.nan,9,5,8,2,6])

x = pd.Series([1, 2, np.nan, np.nan, 5, np.nan, 7, 8, 9, np.nan])


print(ts_moment(series_1, 2, 1))
