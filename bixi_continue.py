
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
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

#from statsmodels.graphics.tsaplots import plot_acf  #自相关图

#1
weather = pd.read_csv("/Users/clairehe/Desktop/bixi/climate.csv")
#weather = weather.loc[312:456,]
weather[['Weather']] = weather [['Weather']].fillna('Cloudy')
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
data_2019_weather['weekday'] = data_2019_weather.start_date.dt.dayofweek
data_2019_weather['num_week'] = data_2019_weather.start_date.dt.weekofyear
data_2019_byhour = data_2019_weather.drop(['start_date','start_station_code', 'end_date', 'end_station_code', 'duration_sec', 'is_member'], axis=1)
data_2019_byhour = data_2019_byhour.groupby(['num_week','weekday','hour']).agg('first')
data_2019_byhour['num_trips'] = data_2019_weather.groupby(['num_week','weekday','hour']).count()['Weather']

f, axes = plt.subplots(6, 1, figsize=(20,15))
plt.sca(axes[0])
sb.countplot(x='weekday',data=data_2019_weather)

#plot the average duration seconds of different hours in a day
plt.sca(axes[1])
sb.countplot(x='hour', data=data_2019_weather)
plt.xlabel('hour of day')
#plot_acf(data_2019_weather['hour'])

#then we need to consider the weather effect
plt.sca(axes[2])
sb.countplot(x='Weather', data=data_2019_weather)
plt.xlabel('weather')


plt.sca(axes[3])
data_filterd = data_2019_weather.drop(['Station Name','Climate ID','Year','Month', 'Day'], axis=1)
corr = data_filterd.corr()
sb.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)



plt.sca(axes[4])

x = data_2019_byhour[['Temp (°C)','Dew Point Temp (°C)','Wind Spd (km/h)']]
y = data_2019_byhour['num_trips']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)

data_2017 = pd.read_csv("/Users/clairehe/Desktop/bixi/OD_2017.csv", encoding="utf-8", low_memory = False)
data_2016 = pd.read_csv("/Users/clairehe/Desktop/bixi/OD_2016.csv", encoding="utf-8", low_memory = False)
data_2015 = pd.read_csv("/Users/clairehe/Desktop/bixi/OD_2015.csv", encoding="utf-8", low_memory = False)
weather_2017 = pd.read_csv("/Users/clairehe/Desktop/bixi/climate2017.csv", encoding="utf-8", low_memory = False)
weather_2016 = pd.read_csv("/Users/clairehe/Desktop/bixi/climate2016.csv", encoding="utf-8", low_memory = False)
weather_2015 = pd.read_csv("/Users/clairehe/Desktop/bixi/climate2015.csv", encoding="utf-8", low_memory = False)

#weather_2017[['Weather']] = weather_2017 [['Weather']].fillna('Cloudy')
weather_2017['Date/Time'] = pd.to_datetime(weather_2017['Date/Time'])
weather_2017['Date/Time'] = weather_2017['Date/Time'].fillna(method='ffill')
weather_2017[['Weather']] = weather_2017 [['Weather']].fillna('Cloudy')
data_2017['start_date'] = pd.to_datetime(data_2017['start_date'], errors ='coerce' )
data_2017['start_date'] = data_2017['start_date'].fillna(method='ffill')
data_2017_weather = data_2017.sort_values(by=['start_date'])
data_2017_weather = data_2017_weather.dropna(axis=1,how='any', thresh=None,subset=None,inplace=False)
data_2017_weather = pd.merge_asof(data_2017_weather, weather_2017, left_on = 'start_date', right_on = 'Date/Time', direction = 'nearest').drop('Date/Time',axis=1)
data_2017_weather['weekday'] = data_2017_weather.start_date.dt.dayofweek
data_2017_weather['hour'] = data_2017_weather.start_date.dt.hour
data_2017_weather['num_week'] = data_2017_weather.start_date.dt.weekofyear
data_2017_weather = data_2017_weather.drop(['start_date','start_station_code', 'end_date', 'end_station_code', 'duration_sec', 'is_member'], axis=1)
data_2017_byhour = data_2017_weather.groupby(['num_week','weekday','hour']).agg('first')
data_2017_byhour['num_trips'] = data_2017_weather.groupby(['num_week','weekday','hour'])['Weather'].count()
data_2017_byhour = data_2017_byhour.dropna(axis=1,how='any', thresh=None,subset=None,inplace=False)

weather_2016[['Weather']] = weather_2016 [['Weather']].fillna('Cloudy')
weather_2016['Date/Time'] = pd.to_datetime(weather_2016['Date/Time'])
weather_2016['Date/Time'] = weather_2016['Date/Time'].fillna(method='ffill')
weather_2016[['Weather']] = weather_2016 [['Weather']].fillna('Cloudy')
data_2016['start_date'] = pd.to_datetime(data_2016['start_date'], errors ='coerce' )
data_2016['start_date'] = data_2016['start_date'].fillna(method='ffill')
data_2016_weather = data_2016.sort_values(by=['start_date'])
data_2016_weather = data_2016_weather.dropna(axis=1,how='any', thresh=None,subset=None,inplace=False)
data_2016_weather = pd.merge_asof(data_2016_weather, weather_2016, left_on = 'start_date', right_on = 'Date/Time', direction = 'nearest').drop('Date/Time',axis=1)
data_2016_weather['weekday'] = data_2016_weather.start_date.dt.dayofweek
data_2016_weather['hour'] = data_2016_weather.start_date.dt.hour
data_2016_weather['num_week'] = data_2016_weather.start_date.dt.weekofyear
data_2016_weather = data_2016_weather.drop(['start_date','start_station_code', 'end_date', 'end_station_code', 'duration_sec', 'is_member'], axis=1)
data_2016_byhour = data_2016_weather.groupby(['num_week','weekday','hour']).agg('first')
data_2016_byhour['num_trips'] = data_2016_weather.groupby(['num_week','weekday','hour'])['Weather'].count()
data_2016_byhour = data_2016_byhour.dropna(axis=1,how='any', thresh=None,subset=None,inplace=False)

weather_2015[['Weather']] = weather_2015 [['Weather']].fillna('Cloudy')
weather_2015['Date/Time'] = pd.to_datetime(weather_2015['Date/Time'])
weather_2015['Date/Time'] = weather_2015['Date/Time'].fillna(method='ffill')
weather_2015[['Weather']] = weather_2015 [['Weather']].fillna('Cloudy')
data_2015['start_date'] = pd.to_datetime(data_2015['start_date'], errors ='coerce' )
data_2015['start_date'] = data_2015['start_date'].fillna(method='ffill')
data_2015_weather = data_2015.sort_values(by=['start_date'])
data_2015_weather = data_2015_weather.dropna(axis=1,how='any', thresh=None,subset=None,inplace=False)
data_2015_weather = pd.merge_asof(data_2015_weather, weather_2015, left_on = 'start_date', right_on = 'Date/Time', direction = 'nearest').drop('Date/Time',axis=1)
data_2015_weather['weekday'] = data_2015_weather.start_date.dt.dayofweek
data_2015_weather['hour'] = data_2015_weather.start_date.dt.hour
data_2015_weather['num_week'] = data_2015_weather.start_date.dt.weekofyear
data_2015_weather = data_2015_weather.drop(['start_date','start_station_code', 'end_date', 'end_station_code', 'duration_sec', 'is_member'], axis=1)
data_2015_byhour = data_2015_weather.groupby(['num_week','weekday','hour']).agg('first')
data_2015_byhour['num_trips'] = data_2015_weather.groupby(['num_week','weekday','hour'])['Weather'].count()
data_2015_byhour = data_2015_byhour.dropna(axis=1,how='any', thresh=None,subset=None,inplace=False)

x = []
y = []


ahead = 24 #24 hours forecast
for input_data in [data_2019_byhour]:
    for k in range(720,input_data.shape[0]-24):
        x.append(np.hstack((
                input_data['num_trips'].values[k-720:k-1], 
                input_data['Visibility (km)'].values[k-24:k+24],    
                input_data['Wind Spd (km/h)'].values[k-24:k+24],   
                input_data['Dew Point Temp (°C)'].values[k-24:k+24],
                input_data['Temp (°C)'].values[k-24:k+24],       
                1                     
            )).tolist()  )
        y.append(input_data['num_trips'].values[k+ahead].tolist() )

x = np.array(x)
y = np.array(y)

#load the past data
x_past = []
y_past = []

                #input_data['Weather'].values[k-24:k+24],           # last month usage

ahead = 24 #24 hours forecast
for input_data in [data_2017_byhour,data_2016_byhour,data_2015_byhour]:
    for k in range(720,input_data.shape[0]-24):
        x_past.append(np.hstack((
                input_data['num_trips'].values[k-720:k-1],           # last month usage
                input_data['Visibility (km)'].values[k-24:k+24],    
                input_data['Wind Spd (km/h)'].values[k-24:k+24],   
                input_data['Dew Point Temp (°C)'].values[k-24:k+24],
                input_data['Temp (°C)'].values[k-24:k+24],                   
                1                     
            )).tolist()  )
        y_past.append(input_data['num_trips'].values[k+ahead].tolist() )

x_past = np.array(x_past)
y_past = np.array(y_past)

#
min_max_scaler = preprocessing.MinMaxScaler()
normalized_x = min_max_scaler.fit_transform(x)
normalized_x_past = min_max_scaler.fit_transform(x_past)

normalized_y = min_max_scaler.fit_transform(y.reshape(-1,1))
normalized_y_past = min_max_scaler.fit_transform(y_past.reshape(-1,1))

clf = linear_model.Ridge (alpha = 1);
clf.fit(normalized_x_past,normalized_y_past);
predict = clf.predict(normalized_x);
score = clf.score(normalized_x,normalized_y)
print('Mean squared error is : {}' .format(mean_squared_error(normalized_y, predict)));
print('the fitness is : {}' .format(score))

plt.sca(axes[5])
plt.plot(normalized_y);
plt.plot(predict,'r');
plt.grid('on');
plt.legend(['Real',u'Prediction']);
plt.axis(xmin=2000,xmax=2720,ymin=-.2,ymax=1.2);
