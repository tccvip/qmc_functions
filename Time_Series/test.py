from TimeSeries_Ops import *
import time


dates1 = pd.date_range(start='2023-01-01', periods=10, freq='D')
dates2 = pd.date_range(start='2023-01-01', periods=9, freq='D')

values = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
series_1 = pd.Series([2,4,4,4,5,5,7,9,5,6], index=dates1)
series_2 = pd.Series(values, index=dates1)
series_3 = pd.Series([1,2,3,4,5,6,7,8,9], index=dates2)
s = pd.Series([5, 5, 6, 7, 5, 5, 5])

df_1 = pd.DataFrame([2,4,4,4,5,5,7,9])
# x = pd.Series([np.nan,9,5,8,2,6])

x = pd.Series([1, '2', np.nan, np.nan, 'nan', np.nan, 7, 8, 9, np.nan])


s = pd.Series([5, 5, 6, 7, 5, 5, 5, 6], index=pd.date_range(start='2023-01-01', periods=8, freq='D'))
t = pd.Series([5, 5, np.nan, 7, 5, 8, 5], index=pd.date_range(start='2023-01-02', periods=7, freq='D'))

