import os
import pandas as pd
import numpy as np
from io import BytesIO
from io import StringIO

from azure.storage.blob import BlobServiceClient, ContainerClient

connect_str = 'DefaultEndpointsProtocol=https;AccountName=avadevblobsapforecast;AccountKey=nAuEl+bhv2xzFsAGqrRkDdnMstQspo+pJ5IicY+YGcFEph6LhwKzjt1++/J36Vk3pBWjBL/0uFXEj/5Sd9XLJA==;EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)  
   
def getFiles(file):
    print(' ----------- Get file names in Blob Storage -----------')
    container = ContainerClient.from_connection_string(connect_str, container_name="datasets")
    blob_list = []
    
    for blob in container.list_blobs():
        if file.replace('*','') in blob.name:
            blob_list.append(blob.name)
    
    return blob_list

def loadData(file_name):
    
    print(' ----------- Load data -----------')
    # connecting to ccai storage
    blob_client = blob_service_client.get_blob_client(container='datasets', blob=file_name)
    
    # Stream all bytes
    content =  blob_client.download_blob()
    stream = BytesIO()
    content.readinto(stream)

    
    
    # Choosing file type
    if '.csv' in file_name:
        s=str(content.readall(),'utf-8')
        strIO = StringIO(s) 
        data = pd.read_csv(strIO)
    elif '.xls' in file_name:
        f = open(file_name, "wb")
        f.write(blob_client.download_blob().content_as_bytes())
        f.close()
        data = pd.read_excel(r''+file_name)
        os.remove(file_name)
    
    return data
 
 
def storeData(results,label,model,TimeColumn,Time_format,Holdout,Resample_frequency,Epoches,Horizon,InputWidth,Shift,SizeTraining,SinCos,select_model_automatically,model_name,statistics,batch,file_name,date_str,FillFuture):
    print(' ----------- Store data -----------')

    if batch:
        file_path = f'run_{date_str}/batch_{file_name}'
    else:
        file_path = f'run_{date_str}'
    
    blob_client = blob_service_client.get_blob_client(container='results', blob=f'{file_path}/predictions.csv')
    blob_client.upload_blob(results.prediction.to_csv(index=False),overwrite=True)
    # os.remove(f'Comparison_{label}_matplot.jpg')
      
    # saving plots
    results.comparison(saveToFile = True, width = 20)
    results.forecast(ratio = 0.01, width = 20, maxTicks = 'auto', saveToFile = True)
    
    blob_client = blob_service_client.get_blob_client(container='results', blob=f'{file_path}/comparison.jpg')
    with open(f'Comparison_{label}_matplot.jpg', "rb") as data:
        blob_client.upload_blob(data,overwrite=True)
    os.remove(f'Comparison_{label}_matplot.jpg')
    
    blob_client = blob_service_client.get_blob_client(container='results', blob=f'{file_path}/forecast.jpg')
    with open(f'Forecast_{label}_matplot.jpg', "rb") as data:
        blob_client.upload_blob(data,overwrite=True)
    os.remove(f'Forecast_{label}_matplot.jpg')
    
    # upload file with settings 
    info_file = os.path.join('info.txt')

    with open(info_file, 'w') as file:
        file.write('Settings:')
        file.write('\nAutomatic Model Selection: {}'.format(select_model_automatically))
        file.write('\nResample Frequency: {}'.format(Resample_frequency))
        file.write('\nTime format: {}'.format(Time_format))
        file.write('\nEpoches: {}'.format(Epoches))
        file.write('\nHorizon: {}'.format(Horizon))
        file.write('\nHoldout: {}'.format(Holdout))
        file.write('\nShift: {}'.format(Shift))
        file.write('\nSize Training: {}'.format(SizeTraining))
        file.write('\nTime Column: {}'.format(TimeColumn))
        file.write('\nSin Cos: {}'.format(SinCos))
        file.write('\nInput Width: {}'.format(InputWidth))
        file.write('\nFill Future: {}'.format(FillFuture))
        file.write('\n-------------------------------')
        file.write('\n\nSelected Model: {}'.format(model_name))
        file.write('\nMAPE after HP-Search: {}%'.format(np.round(model.global_best_precision['mape'],2)))
    
    blob_client = blob_service_client.get_blob_client(container='results', blob=f'{file_path}/info.txt')
    with open(info_file, 'rb') as file:
        blob_client.upload_blob(file,overwrite=True)
    os.remove('info.txt')
        
    # Save Model leads to errors, since different format is needed depending on type of model
    # model.final_model.save("model.h5")
        
    # blob_client = blob_service_client.get_blob_client(container='results', blob=f'{file_path}/model.h5')
    # with open('model.h5', 'rb') as file:
    #     blob_client.upload_blob(file,overwrite=True)
    # os.remove('model.h5')
    
    if statistics:
        blob_client = blob_service_client.get_blob_client(container='results', blob=f'{file_path}/statistics/distplot.jpg')
        with open('distplot.jpg', "rb") as data:
            blob_client.upload_blob(data,overwrite=True)
        os.remove('distplot.jpg')
        
        blob_client = blob_service_client.get_blob_client(container='results', blob=f'{file_path}/statistics/seasonal_decomposition_additive.jpg')
        with open('seasonal_decomposition_additive.jpg', "rb") as data:
            blob_client.upload_blob(data,overwrite=True)
        os.remove('seasonal_decomposition_additive.jpg')
        
        if 'seasonal_decomposition_multiplicative.jpg' in os.listdir(): #if negative values, no multiplicative image possible
            blob_client = blob_service_client.get_blob_client(container='results', blob=f'{file_path}/statistics/seasonal_decomposition_multiplicative.jpg')
            with open('seasonal_decomposition_multiplicative.jpg', "rb") as data:
                blob_client.upload_blob(data,overwrite=True)
            os.remove('seasonal_decomposition_multiplicative.jpg')
        
        blob_client = blob_service_client.get_blob_client(container='results', blob=f'{file_path}/statistics/regression_trend.jpg')
        with open('regression_trend.jpg', "rb") as data:
            blob_client.upload_blob(data,overwrite=True)
        os.remove('regression_trend.jpg')
        
        blob_client = blob_service_client.get_blob_client(container='results', blob=f'{file_path}/statistics/Statistics.txt')
        with open('Statistics.txt', 'rb') as file:
            blob_client.upload_blob(file,overwrite=True)
        os.remove('Statistics.txt')

    print(' ----------- Upload successfull -----------')
    print(f'Run id: {date_str}')
    return 'run_' + str(date_str)