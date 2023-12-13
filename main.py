from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import pymysql
import joblib
import uvicorn
from sklearn.datasets import load_iris
from datetime import datetime
from datetime import date

app = FastAPI()
# https://drive.google.com/file/d/1BXmIQiBVHZnWmqZkVtLE3dDxScGMHAps/view?usp=sharing

class form_data(BaseModel):
    vin : str
    partCode : str
    vehicleModel : str
    currentMileage : float
    decDate : str
    vehicleOwners : float
    numWo : int
    customerType : str

# class request_body(BaseModel):
#     sepal_length : int
#     sepal_widht : int
#     petal_length : int
#     petal_widht : int

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Endpoint to receive POST request for prediction
@app.post('/predict')
async def predict(data : form_data):
    try:
        battery_list = ['28800YZZNJ', '28800YZZNH', '28800YZZWS', '28800YZZH1', '28800YZZWR', '28800YZZDH', '28800YZZWQ', '28800YZZNG', '28800YZZH2', '28800YZZKL', '28800YZZLN', '28800YZZFH', '28800YZZH3', '28800YZZXK', '28800YZZDK', '28800YZZNZ', '28800YZZH4', '28800YZZDM', '28800YZZH8', '28800YZZH5', '288000T131', '288000U010', '288000C210', '2880067131', '28800YZZH7', '288000C273', '288000L502', '288000T140', '2880028100']
        battery_type_list = [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        index = battery_list.index(data.partCode)
        batteryType = battery_type_list[index]

        dDate = datetime.strptime(data.decDate, '%Y-%m-%d')
        age = (datetime.now() - dDate).days
        if age > (365*8) :
            ageCategory = '> 8 tahun'
        elif age > (365*7) : 
            ageCategory = '7-8 tahun'
        elif age > (365*6) : 
            ageCategory = '6-7 tahun'
        elif age > (365*5) : 
            ageCategory = '5-6 tahun'
        elif age > (365*4) : 
            ageCategory = '4-5 tahun'
        elif age > (365*3) : 
            ageCategory = '3-4 tahun'
        elif age > (365*2) : 
            ageCategory = '2-3 tahun'
        elif age > (365*1) : 
            ageCategory = '1-2 tahun'
        else: 
            ageCategory = '< 1 tahun'

        test_data = pd.DataFrame({
            'Vehicle Model' : [data.vehicleModel],
            'Mileage' : [data.currentMileage],
            'Battery Type' : [batteryType],
            'age_category' : [ageCategory],
            'Penggantian' : [data.numWo],
            'Banyak Kendaraan' : [data.vehicleOwners],
            'CustomerType' : [data.customerType]
        })

        model = joblib.load('./models/my_model.pkl')
        result = model.predict(test_data)[0]

        return {'result' : result}

    except Exception as e:
        return {'error': str(e)}

# @app.post('/test')
# async def predict(data : request_body):
#     test_data = [[
#             data.sepal_length, 
#             data.sepal_width, 
#             data.petal_length, 
#             data.petal_width
#     ]]
#     iris = load_iris()
#     clf = joblib.load('./models/model.pkl')
#     class_idx = clf.predict(test_data)[0]
#     return { 'class' : iris.target_names[class_idx]}

port = 8889
if __name__ == "__main__":
   uvicorn.run(app, host="127.0.0.1", port=port)
