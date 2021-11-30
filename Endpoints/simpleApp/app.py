from flask import Flask, request
from flask import jsonify
import random, string
from steps.training import trainingStep
from steps.azurestorage import loadData, storeData
from utilities.deleteblob import deletefiles

app = Flask(__name__)

@app.before_request
def before_func():
	print("Welcome to your SAP Forecasting Service!")

@app.route('/training', methods=['GET'])
def training():
    # Parse Argumenst
	filename = request.args.get('filename')
    
	# Load Data from Azure Storage
	data = loadData(filename)
    
    # Preprocessing and Training
	trainingResult = trainingStep(data)
 
	# Store Results in Azure Storage
	run_id = storeData(trainingResult)
 
	return jsonify(run_id), 200


@app.route('/inference', methods=['GET'])
def inference(): 
	return jsonify('here should be some inference stuff happening'), 200


@app.route('/deleteblobs', methods=['GET'])
def deleteblobs(): 
    deletefiles()
    return jsonify('Successfully deleted all files!'), 200

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000)