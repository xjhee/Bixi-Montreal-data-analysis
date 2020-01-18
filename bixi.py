import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb
import math
from datetime import datetime
import datetime as dt

a = pd.read_csv("/Users/clairehe/Desktop/bixi/OD_2017.csv", encoding="utf-8", low_memory = False)
station_2019 = pd.read_csv("/Users/clairehe/Desktop/bixi/Stations_2019.csv", encoding="utf-8", low_memory = False)
data_2019 = pd.read_csv("/Users/clairehe/Desktop/bixi/merged2019.csv", encoding="utf-8", low_memory = False)
data_2019['start_station_code'] = data_2019['start_station_code'].convert_objects(convert_numeric = True)
data_2019['start_station_code'] = data_2019['start_station_code'].fillna(0).astype(np.int64)

data_2019['end_station_code'] = data_2019['end_station_code'].convert_objects(convert_numeric = True)
data_2019['end_station_code'] = data_2019['end_station_code'].fillna(0).astype(np.int64)

data_2019['duration_sec'] = data_2019['duration_sec'].convert_objects(convert_numeric = True)
data_2019['duration_sec'] = data_2019['duration_sec'].fillna(0).astype(np.int64)

data_2019['is_member'] = data_2019['is_member'].convert_objects(convert_numeric = True)
data_2019['is_member'] = data_2019['is_member'].fillna(0).astype(np.int64)


#we first start to analyze the data by finding and graphing out the trend of riding in monthly data and in day of week
'''date = data_2019[['start_date', 'end_date']]
day = pd.DatetimeIndex(date['start_date']).day
sb.countplot(day)
plt.xlabel('Day of the month')'''


#to further compare the usage of each station, we must sort out the total duration time and the according counts
#first we sort out the total duration seconds with the first col start station code
sort_start_sec = pd.DataFrame(data_2019.groupby(by=['start_station_code'])['duration_sec'].sum())
sort_start_sec.columns = ['Total duration seconds']
sort_start_sec = sort_start_sec.sort_values(by = 'Total duration seconds', ascending=False)
#sort_start.head()

#then we sort out the total used time
sort_start_count = data_2019.groupby(by=['start_station_code'])['start_date'].agg({'Count': np.size})
sort_start_count['Count'] = sort_start_count.Count
sort_start_count = sort_start_count.sort_values(by = 'Count', ascending=False)
#sort_start_count.head()

sort_end_count = data_2019.groupby(by=['end_station_code'])['end_date'].agg({'Count': np.size})
sort_end_count['Count'] = sort_end_count.Count
sort_end_count = sort_end_count.sort_values(by = 'Count', ascending=False)

#we now try to investigate the TOP used station; 
#we first name out the most used station both through duration time and count times
most_used_station_s = sort_start_sec.index[1]   
most_used_station_c = sort_start_count.index[1]
print('The most used station is numbered: {} based on duration time and is numbered :{} based on total used times'.format(most_used_station_s,most_used_station_c))

#now we plot the TOP20 duration and TOP20 counts
f, axes = plt.subplots(2, 1, figsize=(15,10))
plt.sca(axes[0])
top = np.array(sort_start_sec.head(20).index)
topd = data_2019[data_2019['start_station_code'].isin(top)]
sb.lvplot(data=topd,x='start_station_code', y='duration_sec',order=top)
plt.title('The Top20 used station by duration in 2019', fontsize = 18)

plt.sca(axes[1])
top_data_s = np.array(sort_start_count.head(20).index)
topd_data_s = data_2019[data_2019['start_station_code'].isin(top_data_s)]
sb.countplot(data=topd_data_s,x='start_station_code',order=top_data_s)
plt.title('The Top20 used station to start at by counts in 2019', fontsize = 18)

plt.sca(axes[1])
top_data_e = np.array(sort_end_count.head(20).index)
topd_data_e = data_2019[data_2019['end_station_code'].isin(top_data_e)]
sb.countplot(data=topd_data_e,x='end_station_code',order=top_data_e)
plt.title('The Top20 used station to end at by counts in 2019', fontsize = 18)

sum_top_data_s = np.array(sort_start_count.head(20).sum())
sum_top_data_e = np.array(sort_end_count.head(20).sum())
ratio = (sum_top_data_s + sum_top_data_e)/ data_2019.shape[0]
print('Sum of the top 20 stations rentals to start at: {}'.format(sum_top_data_s))
print('Sum of the top 20 stations rentals to end at: {}'.format(sum_top_data_e))
print('Sum of both start and end station rentals for top 20 stations: {}'.format(sum_top_data_s + sum_top_data_e))
print('This is {} of total rentals for 2019'.format(ratio))

#we then try to find the least used station through two ways
least_used_station_s = sort_start_sec.index[-1] 
least_used_station_c = sort_start_count.index[-1]
print('The leaset used station is numbered: {} based on duration time and is numbered :{} based on total used times'.format(least_used_station_s,least_used_station_c))
#now we plot the least used station information
f, axes = plt.subplots(2, 1, figsize=(15,10))
plt.sca(axes[0])
l = np.array(sort_start_sec.tail(20).index)
ld = data_2019[data_2019['start_station_code'].isin(l)]
sb.lvplot(data=ld,x='start_station_code', y='duration_sec',order=l)
plt.title('The 20 least used station by duration in 2019', fontsize = 18)

plt.sca(axes[1])
l_data_s = np.array(sort_start_count.tail(20).index)
ld_data_s = data_2019[data_2019['start_station_code'].isin(l_data_s)]
sb.countplot(data=ld_data_s,x='start_station_code',order=l_data_s)
plt.title('The 20 leaset used station to start at by counts in 2019', fontsize = 18)

sum_l_data_s = np.array(sort_start_count.tail(20).sum())
sum_l_data_e = np.array(sort_end_count.tail(20).sum())
ratio2 = (sum_l_data_s + sum_l_data_e)/ data_2019.shape[0]
print('Sum of the least used 20 stations for rentals to start at: {}'.format(sum_l_data_s))
print('Sum of the least used 20 stations for rentals to end at: {}'.format(sum_l_data_e))
print('Sum of both start and end station rentals for least used 20 stations: {}'.format(sum_l_data_s + sum_l_data_e))
print('This is {} of total rentals for 2019'.format(ratio2))






