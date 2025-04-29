import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf 
import pickle

df = pd.read_pickle('data/weather_pandas.pkl')
scaler = MinMaxScaler()
num_m = ['temp', 'dew_point', 'humidity', 'precipitation', 'wind_speed', 'pressure']
#need to norm num metrics 
df[num_m] = scaler.fit_transform(df[num_m])
#this will convert string to of type date
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayw'] = df['date'].dt.dayofweek
#get_dummies taking in all vals of day and 
dayohe = pd.get_dummies(df['dayw'])
df = pd.concat([df, dayohe], axis=1)
df = df.drop('dayw', axis=1)
df = df.drop('date', axis=1)

weather_ten = tf.convert_to_tensor(df.values, dtype=tf.float32)
with open('data/weather_pandas.pkl', 'wb') as f:  
    pickle.dump(df, f)
    print('pandas pickle dumped')
with open('data/weather_tensor.pkl', 'wb') as f:  
    pickle.dump(weather_ten, f)
    print('tensor pickle dumped')



