# Bixi-Montreal-data-analysis
analysis on various variables' effect on number of bike-rent trips 
## A. we need first to process data in a code readable mean 
here we start to analyze the data from year 2019
```
#read data and then assign to match data type
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
#then we sort out the total duration seconds with the first col start station code
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

```
## B. To analyze the data we need to answer the following questions:
## 1). Generally, what are the most frequently used station in Montreal and what are the least
```
#we now try to investigate the TOP used station; 
#we first name out the most used station both through duration time and count times
most_used_station_s = sort_start_sec.index[1]   
most_used_station_c = sort_start_count.index[1]
print('The most used station is numbered: {} based on duration time and is numbered :{} based on total used times'.format(most_used_station_s,most_used_station_c))
```
![estimator](https://tva1.sinaimg.cn/large/006tNbRwgy1gb1bvfxxw5j30wc02qwew.jpg)

### plot the TOP20 station by duration and by counts
```
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

```
![estimator](https://tva1.sinaimg.cn/large/006tNbRwgy1gb1c7yxkt0j30x00m80x2.jpg)

### the leaset used 20 stations 
```
#we then try to find the least used station through two ways then plot the according graph
least_used_station_s = sort_start_sec.index[-1]  
least_used_station_c = sort_start_count.index[-1]
print('The leaset used station is numbered: {} based on duration time and is numbered :{} based on total used times'.format(least_used_station_s,least_used_station_c))

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
```
![estimator](https://tva1.sinaimg.cn/large/006tNbRwgy1gb1ck6l7cmj31720u0ag3.jpg)

### 2). What is the general proportion of the most used station (ie, what proportion of the TOP20 stations and the TAIL20 stations to the whole stations)
```
sum_top_data_s = np.array(sort_start_count.head(20).sum())
sum_top_data_e = np.array(sort_end_count.head(20).sum())
ratio = (sum_top_data_s + sum_top_data_e)/ data_2019.shape[0]
print('Sum of the top 20 stations rentals to start at: {}'.format(sum_top_data_s))
print('Sum of the top 20 stations rentals to end at: {}'.format(sum_top_data_e))
print('Sum of both start and end station rentals for top 20 stations: {}'.format(sum_top_data_s + sum_top_data_e))
print('This is {} of total rentals for 2019'.format(ratio))

sum_l_data_s = np.array(sort_start_count.tail(20).sum())
sum_l_data_e = np.array(sort_end_count.tail(20).sum())
ratio2 = (sum_l_data_s + sum_l_data_e)/ data_2019.shape[0]
print('Sum of the least used 20 stations for rentals to start at: {}'.format(sum_l_data_s))
print('Sum of the least used 20 stations for rentals to end at: {}'.format(sum_l_data_e))
print('Sum of both start and end station rentals for least used 20 stations: {}'.format(sum_l_data_s + sum_l_data_e))
print('This is {} of total rentals for 2019'.format(ratio2))
```
![estimator](https://tva1.sinaimg.cn/large/006tNbRwgy1gb1cdunka7j31ay06kmzi.jpg)

![estimator](https://tva1.sinaimg.cn/large/006tNbRwgy1gb1cmzswuaj319q04ywg7.jpg)

### 3). What is the week/day/hour effect on the number of trips 
we need to calculate the number of trips through indexes in hour,day and week and then plot according indexes
we can draw conclusion from the graph that people tend to go on more trips on weekdays and they tend to have the most frequent bike rental during commuting time
```
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


```
![estimator](https://tva1.sinaimg.cn/large/006tNbRwgy1gb1ctxekbwj318w0fswg2.jpg)

### 4). Draw a heatmap of all attributes to show their relative influence on each other
```
plt.sca(axes[3])
data_filterd = data_2019_weather.drop(['Station Name','Climate ID','Year','Month', 'Day'], axis=1)
corr = data_filterd.corr()
sb.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
```
![estimator](https://tva1.sinaimg.cn/large/006tNbRwgy1gb1cwolr2uj313y0540tv.jpg)

### 5). Draw a linear regression graph between ‘temperature’, ‘Dew Point Temp’, ‘Wind Spead’ and number of trips
```
plt.sca(axes[4])
x = data_2019_byhour[['Temp (°C)','Dew Point Temp (°C)','Wind Spd (km/h)']]
y = data_2019_byhour['num_trips']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
```
![estimator](https://tva1.sinaimg.cn/large/006tNbRwgy1gb1cyfoqa0j316g04udh1.jpg)

### 6). Load past data from year 2015-2017, apply ridge regression to estimate and predict future trend
#### first we process the data from 2015-2017  
```
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

```
#### then we create dynamic regressor for test data 
```
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

```
#### then we use minmaxscaler to scale each feature to a given range
```
min_max_scaler = preprocessing.MinMaxScaler()
normalized_x = min_max_scaler.fit_transform(x)
normalized_x_past = min_max_scaler.fit_transform(x_past)

normalized_y = min_max_scaler.fit_transform(y.reshape(-1,1))
normalized_y_past = min_max_scaler.fit_transform(y_past.reshape(-1,1))

```
#### then we use clf ridge to estimate and predict future trend
```
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

```
![estimator](https://tva1.sinaimg.cn/large/006tNbRwgy1gb1d6n2i42j314a02g74o.jpg)
![estimator](https://tva1.sinaimg.cn/large/006tNbRwgy1gb1d6512eej316805e0wd.jpg)

