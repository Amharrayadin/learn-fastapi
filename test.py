import joblib
import pandas as pd

battery = pd.read_pickle('./datasets/battery-processed-(CustomerType,Banyak Kendaraan).pkl')
bins = [0, 365*1, 365*2, 365*3, 365*4, 365*5, 365*6, 365*7, 365*8, battery['age'].max()]
labels = ['< 1 tahun', '1-2 tahun', '2-3 tahun', '3-4 tahun', '4-5 tahun', '5-6 tahun', '6-7 tahun', '7-8 tahun', '> 8 tahun']
battery['age_category'] = pd.cut(battery['age'], bins=bins, labels=labels)
features = ['Vehicle Model', 'Mileage', 'Battery Type', 'age_category', 'Penggantian', 'Banyak Kendaraan', 'CustomerType']
X = battery[features]

model = joblib.load('./models/my_model.pkl')

result = model.predict(X)

print(result)