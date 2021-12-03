import os, datetime
from flask import Flask, request
from flask import jsonify
import random, string
from steps.exploration import exploration
from steps.training import trainingStep
from steps.azurestorage import loadData, storeData, getFiles
from utilities.deleteblob import deletefiles
from utilities.helper import boolConv

app = Flask(__name__)

@app.before_request
def before_func():
	print("Welcome to your SAP Forecasting Service!")

@app.route('/training', methods=['GET'])
def training():
	# Parse Argumenst
	filename=request.args.get('filename')
	timecolumn=request.args.get('timecolumn')
	timeformat=request.args.get('timeformat')
	labelColumn=request.args.get('labelColumn')#if aggregating then add _aggregationMethod (eg. "sales_sum")
	holdout=int(request.args.get('holdout',default=100))
	resamplefrequency = request.args.get('resamplefrequency',default='D')
	epoches = int(request.args.get('epoches',default=20))
	horizon = int(request.args.get('horizon',default=holdout))
	inputwidth = int(request.args.get('inputWidth',default=2))
	shift = int(request.args.get('shift',default=1))
	sizetraining = float(request.args.get('sizeTraining',default=0.7))
	sincos = boolConv(request.args.get('sincos',default='False'))
	modeldetection = boolConv(request.args.get('modeldetection',default='True'))
	statistics = boolConv(request.args.get('statistics',default='True'))
	batch = boolConv(request.args.get('batch',default='False'))
	fillfuture = boolConv(request.args.get('fillfuture',default='False'))
 
	if filename!=None and timecolumn!=None and timecolumn!=None and labelColumn!=None:
		now = datetime.datetime.now()
		date_str = now.strftime('%Y%m%d%H%M%S')

		if batch:
			# Simple For Loop for multiple files: Currently must have same settings (same label, same time column, epoches must be same etc.)
			filelist = getFiles(filename)
			print(filelist)
			for file in filelist:
				data=loadData(file)	

				# Explorative Analysis
				if statistics:
					exploration(data[[timecolumn,labelColumn]],timecolumn,labelColumn,resamplefrequency)
					print('Statistics created!')
     
				# Preprocessing and Training
				trainingResult, model, model_name = trainingStep(data,timecolumn,timeformat,labelColumn,holdout,resamplefrequency,epoches,horizon,inputwidth,shift,sizetraining,sincos,modeldetection,fillfuture)
				print('Training finished!')
    
				# # Store Results in Azure Storage
				run_id = storeData(trainingResult,labelColumn,model,timecolumn,timeformat,holdout,resamplefrequency,epoches,horizon,inputwidth,shift,sizetraining,sincos,modeldetection,model_name,statistics,batch,file,date_str,fillfuture)

			if len(filelist) != 0:
				return jsonify(run_id), 200
			else:
				return jsonify('Check your uploaded data: Possible issues is incorrect file name or no such file in blob storage'), 500
		else:
			# loading data from azure blob storage
			data=loadData(filename)	

			# Explorative Analysis
			if statistics:
				exploration(data[[timecolumn,labelColumn]],timecolumn,labelColumn,resamplefrequency)
	
			# Preprocessing and Training
			trainingResult, model, model_name = trainingStep(data,timecolumn,timeformat,labelColumn,holdout,resamplefrequency,epoches,horizon,inputwidth,shift,sizetraining,sincos,modeldetection,fillfuture)
	
			# # Store Results in Azure Storage
			run_id = storeData(trainingResult,labelColumn,model,timecolumn,timeformat,holdout,resamplefrequency,epoches,horizon,inputwidth,shift,sizetraining,sincos,modeldetection,model_name,statistics,batch,filename,date_str,fillfuture)

			return jsonify(run_id), 200
	else:
		return jsonify('Missing required input parameters'), 500


@app.route('/inference', methods=['GET'])
def inference(): 
	return jsonify('here should be some inference stuff happening'), 200


@app.route('/deleteblobs', methods=['GET'])
def deleteblobs(): 
    deletefiles()
    return jsonify('Successfully deleted all files!'), 200

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)