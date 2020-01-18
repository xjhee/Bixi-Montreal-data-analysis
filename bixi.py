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
least_used_station_s = sort_start_sec.index[-2]   #记得把它改成-1哈
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






--------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb
import math
from datetime import datetime
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#from statsmodels.graphics.tsaplots import plot_acf  #自相关图

#1
weather = pd.read_csv("/Users/clairehe/Desktop/bixi/climate.csv")
#weather = weather.loc[312:456,]
weather[['Weather']] = weather [['Weather']].fillna('Non extreme weather')
avg_tem_hourly = weather.groupby('Date/Time')['Temp (°C)'].sum()
avg_wind_speed_hourly = weather.groupby('Date/Time')['Wind Spd (km/h)'].sum()
avg_visibility_hourly = weather.groupby('Date/Time')['Visibility (km)'].sum()

#2
data_2019 = pd.read_csv("/Users/clairehe/Desktop/bixi/merged2019.csv", encoding="utf-8", low_memory = False)
data_2019['start_station_code'] = data_2019['start_station_code'].convert_objects(convert_numeric = True)
data_2019['start_station_code'] = data_2019['start_station_code'].fillna(0).astype(np.int64)

data_2019['end_station_code'] = data_2019['end_station_code'].convert_objects(convert_numeric = True)
data_2019['end_station_code'] = data_2019['end_station_code'].fillna(0).astype(np.int64)

data_2019['duration_sec'] = data_2019['duration_sec'].convert_objects(convert_numeric = True)
data_2019['duration_sec'] = data_2019['duration_sec'].fillna(0).astype(np.int64)

data_2019['is_member'] = data_2019['is_member'].convert_objects(convert_numeric = True)
data_2019['is_member'] = data_2019['is_member'].fillna(0).astype(np.int64)

weather['Date/Time'] = pd.to_datetime(weather['Date/Time'])
weather['Date/Time'] = weather['Date/Time'].fillna(method='ffill')
data_2019['start_date'] = pd.to_datetime(data_2019['start_date'], errors ='coerce' )
data_2019['start_date'] = data_2019['start_date'].fillna(method='ffill')
data_2019_weather = data_2019.sort_values(by=['start_date'])
data_2019_weather = pd.merge_asof(data_2019_weather, weather, left_on = 'start_date', right_on = 'Date/Time', direction = 'nearest').drop('Date/Time',axis=1)
data_2019_weather = data_2019_weather.dropna(axis=1,how='any', thresh=None,subset=None,inplace=False)

#plot the average frequency within a week
data_2019_weather['hour'] = data_2019_weather.start_date.dt.hour
data_2019_weather['day'] = data_2019_weather.start_date.dt.dayofweek
data_2019_weather['week'] = data_2019_weather.start_date.dt.weekofyear
data_2019_weather = data_2019_weather.drop(['start_date','start_station_code', 'end_date', 'end_station_code', 'duration_sec', 'is_member', 'Temp (°C)'], axis=1)
data_2019_byhour = data_2019_weather.groupby(['week','day','hour']).agg('first')
data_2019_byhour['num_trips'] = data_2019_byhour.groupby(['week','day','hour']).count()['Weather']

#f, axes = plt.subplots(5, 1, figsize=(20,15))
#plt.sca(axes[0])
#sb.countplot(x='day',data=data_2019_weather)

#plot the average duration seconds of different hours in a day
#plt.sca(axes[1])
#sb.countplot(x='hour', data=data_2019_weather)
#plt.xlabel('hour of day')
#plot_acf(data_2019_weather['hour'])

#then we need to consider the weather effect
#plt.sca(axes[2])
#sb.countplot(x='Weather', data=data_2019_weather)
#plt.xlabel('weather')


#plt.sca(axes[3])
#data_filterd = data_2019_weather.drop(['Station Name','Climate ID','Year','Month', 'Day'], axis=1)
#corr = data_filterd.corr()
#sb.heatmap(corr, 
        #xticklabels=corr.columns,
       # yticklabels=corr.columns)



#plt.sca(axes[4])

#x = data_2019_weather[['Temp (°C)','Dew Point Temp (°C)','Wind Spd (km/h)']]
#y = data_2019_weather['duration_sec']
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
#lm = LinearRegression()
#lm.fit(X_train,y_train)
#predictions = lm.predict(X_test)
#plt.scatter(y_test,predictions)

data_2017 = pd.read_csv("/Users/clairehe/Desktop/bixi/OD_2017.csv", encoding="utf-8", low_memory = False)
data_2016 = pd.read_csv("/Users/clairehe/Desktop/bixi/OD_2016.csv", encoding="utf-8", low_memory = False)
data_2015 = pd.read_csv("/Users/clairehe/Desktop/bixi/OD_2015.csv", encoding="utf-8", low_memory = False)
weather_2017 = pd.read_csv("/Users/clairehe/Desktop/bixi/climate2017.csv", encoding="utf-8", low_memory = False)
weather_2016 = pd.read_csv("/Users/clairehe/Desktop/bixi/climate2016.csv", encoding="utf-8", low_memory = False)
weather_2015 = pd.read_csv("/Users/clairehe/Desktop/bixi/climate2015.csv", encoding="utf-8", low_memory = False)

weather_2017['Date/Time'] = pd.to_datetime(weather_2017['Date/Time'])
weather_2017['Date/Time'] = weather_2017['Date/Time'].fillna(method='ffill')
data_2017['start_date'] = pd.to_datetime(data_2017['start_date'], errors ='coerce' )
data_2017['start_date'] = data_2017['start_date'].fillna(method='ffill')
data_2017_weather = data_2017.sort_values(by=['start_date'])
data_2017_weather = data_2017_weather.dropna(axis=1,how='any', thresh=None,subset=None,inplace=False)
data_2017_weather = pd.merge_asof(data_2017_weather, weather_2017, left_on = 'start_date', right_on = 'Date/Time', direction = 'nearest').drop('Date/Time',axis=1)
data_2017_weather['weekday_2017'] = data_2017_weather.start_date.dt.dayofweek
data_2017_weather['hour_2017'] = data_2017_weather.start_date.dt.hour
data_2017_weather['num_week_2017'] = data_2017_weather.start_date.dt.weekofyear
data_2017_weather = data_2017_weather.drop(['start_date','start_station_code', 'end_date', 'end_station_code', 'duration_sec', 'is_member', 'Temp (°C)'], axis=1)
data_2017_byhour = data_2017_weather.groupby(['num_week_2017','weekday_2017','hour_2017']).agg('first')
data_2017_byhour['num_trips_2017'] = data_2017_byhour.groupby(['num_week_2017','weekday_2017','hour_2017'])['Weather'].count()

weather_2016['Date/Time'] = pd.to_datetime(weather_2016['Date/Time'])
weather_2016['Date/Time'] = weather_2016['Date/Time'].fillna(method='ffill')
data_2016['start_date'] = pd.to_datetime(data_2016['start_date'], errors ='coerce' )
data_2016['start_date'] = data_2016['start_date'].fillna(method='ffill')
data_2016_weather = data_2016.sort_values(by=['start_date'])
data_2016_weather = data_2016_weather.dropna(axis=1,how='any', thresh=None,subset=None,inplace=False)
data_2016_weather = pd.merge_asof(data_2016_weather, weather_2016, left_on = 'start_date', right_on = 'Date/Time', direction = 'nearest').drop('Date/Time',axis=1)
data_2016_weather['weekday_2016'] = data_2016_weather.start_date.dt.dayofweek
data_2016_weather['hour_2016'] = data_2016_weather.start_date.dt.hour
data_2016_weather['num_week_2016'] = data_2016_weather.start_date.dt.weekofyear
data_2016_weather = data_2016_weather.drop(['start_date','start_station_code', 'end_date', 'end_station_code', 'duration_sec', 'is_member', 'Temp (°C)'], axis=1)
data_2016_byhour = data_2016_weather.groupby(['num_week_2016','weekday_2016','hour_2016']).agg('first')
data_2016_byhour['num_trips_2016'] = data_2016_byhour.groupby(['num_week_2016','weekday_2016','hour_2016'])['Weather'].count()

weather_2015['Date/Time'] = pd.to_datetime(weather_2015['Date/Time'])
weather_2015['Date/Time'] = weather_2015['Date/Time'].fillna(method='ffill')
data_2015['start_date'] = pd.to_datetime(data_2015['start_date'], errors ='coerce' )
data_2015['start_date'] = data_2015['start_date'].fillna(method='ffill')
data_2015_weather = data_2015.sort_values(by=['start_date'])
data_2015_weather = data_2015_weather.dropna(axis=1,how='any', thresh=None,subset=None,inplace=False)
data_2015_weather = pd.merge_asof(data_2015_weather, weather_2015, left_on = 'start_date', right_on = 'Date/Time', direction = 'nearest').drop('Date/Time',axis=1)
data_2015_weather['weekday_2015'] = data_2015_weather.start_date.dt.dayofweek
data_2015_weather['hour_2015'] = data_2015_weather.start_date.dt.hour
data_2015_weather['num_week_2015'] = data_2015_weather.start_date.dt.weekofyear
data_2015_weather = data_2015_weather.drop(['start_date','start_station_code', 'end_date', 'end_station_code', 'duration_sec', 'is_member', 'Temp (°C)'], axis=1)
data_2015_byhour = data_2015_weather.groupby(['num_week_2015','weekday_2015','hour_2015']).agg('first')
data_2015_byhour['num_trips_2015'] = data_2015_byhour.groupby(['num_week_2015','hour_2015','num_week_2015'])['Weather'].count()

Xtst = []
ytst = []

Ntst = data_2019_byhour.shape[0]

ahead = 24 #24 hours forecast
for k in range(720,Ntst-24):
    Xtst.append(np.hstack((
                data_2019_byhour['num_trips'].values[k-720:k-1],           # last month usage
                data_2019_byhour['Weather'].values[k-24:k+24],           # last month usage
                data_2019_byhour['Visibility (km)'].values[k-24:k+24],    
                data_2019_byhour['Wind Spd (km/h)'].values[k-24:k+24],   
                data_2019_byhour['Dew Point Temp (°C)'].values[k-24:k+24],
                data_2019_byhour['Temp (°C)'].values[k-24:k+24],              
                data_2019_byhour['Time'].values[k],                 
                1                     
            )).tolist()  )
    ytst.append(data_2019_weather['num_trips'].values[k+ahead].tolist() )

Xtst = np.array(Xtst)
ytst = np.array(ytst)