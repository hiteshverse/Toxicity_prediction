from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

##Load the Models
from utils import *
from main import *

# Initialize Flask
app=Flask(__name__)

### create the function
@app.route('/Predict',methods=['POST'])
def predict_toxicity():
    try:
        data = request.get_json()
        smi_list = data['smiles']
        toxicity_name = data['toxicity_name']  #['Hepatotoxicity','Mutagenicity','Cardiotoxicity','Carcinogenicity','Nephrotoxicity']

        # Load models based on the toxicity names
        model_directory = 'models'  # Directory where your models are stored
        new_dict = utils.LOAD_MODEL(model_directory, toxicity_name)

        # Generate predictions
        df = ALL_MODEL_PREDICTION(smi_list, new_dict)

        return df.to_json(orient='records')  # Convert DataFrame to JSON and return
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Define a simple health check endpoint
@app.route('/')
def health_check():
    return "Toxicity Prediction API is running"


if __name__=="__main__":
    app.run(debug=True,port=5001)  # Change port number to 5001 or any other available port