import os
import uuid
import pandas as pd
import datetime

from io import BytesIO
from io import StringIO

from azure.storage.blob import BlobServiceClient

connect_str = 'DefaultEndpointsProtocol=https;AccountName=avadevblobsapforecast;AccountKey=nAuEl+bhv2xzFsAGqrRkDdnMstQspo+pJ5IicY+YGcFEph6LhwKzjt1++/J36Vk3pBWjBL/0uFXEj/5Sd9XLJA==;EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)  
   
def loadData(file_name):
    
    # # Get parameters
    # print(os.getcwd())
    # path = './data/data.csv'
    
    # # read data file
    # data = pd.read_csv(path, engine='python')
    
    print(' ----------- Load data -----------')
    # connecting to ccai storage
    blob_client = blob_service_client.get_blob_client(container='datasets', blob=file_name)
    
    # Stream all bytes
    content =  blob_client.download_blob()
    stream = BytesIO()
    content.readinto(stream)

    s=str(content.readall(),'utf-8')
    strIO = StringIO(s) 
    data = pd.read_csv(strIO)
    
    return data
 
 
def storeData(results):
    print(' ----------- Store data -----------')
    now = datetime.datetime.now()
    date_str = now.strftime('%Y%m%d%H%M%S')
    
    blob_client = blob_service_client.get_blob_client(container='results', blob=f'run_{date_str}/predictions.csv')
    blob_client.upload_blob(results.prediction.to_csv(index=False),overwrite=True)
    
    # saving plots
    results.comparison(saveToFile = True, width = 20)
    results.forecast(ratio = 0.01, width = 20, maxTicks = 'auto', saveToFile = True)
    
    blob_client = blob_service_client.get_blob_client(container='results', blob=f'run_{date_str}/comparison.jpg')
    with open('Comparison_sales_matplot.jpg', "rb") as data:
        blob_client.upload_blob(data,overwrite=True)
    
    blob_client = blob_service_client.get_blob_client(container='results', blob=f'run_{date_str}/forecast.jpg')
    with open('Forcast_sales_matplot.jpg', "rb") as data:
        blob_client.upload_blob(data,overwrite=True)
    
    # blob_client = blob_service_client.get_blob_client(container='results', blob=f'run_{date_str}/settings.txt')
    # blob_client.upload_blob(results.prediction.to_csv(index=False),overwrite=True)

    print(' ----------- Upload successfull -----------')
    print(f'Run id: {date_str}')
    return 'run_' + str(date_str)