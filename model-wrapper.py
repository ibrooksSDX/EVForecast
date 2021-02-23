import sys
import cdsw
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import pickle

# == For Testing ==
features = ['model_C_sales', 'model_D_sales', 'model_R_sales',
       'part_no']
args = {  "model_C_sales" : "40",
  "model_D_sales" : "82",
  "model_R_sales" : "34",
  "part_no" : 'a42CLDR'
}

# == Main Function ==
def PredictFunc(args):
	# Load Data
	filtArgs = {key: [args[key]] for key in features}
	data = pd.DataFrame.from_dict(filtArgs)

	# Load Model
	with open(args["part_no"] + '.pickle', 'rb') as handle:
		mdl = pickle.load(handle)
	model = pickle.loads(mdl)

	# Get Prediction
	prediction = model.predict(data)
  
  # Return Prediction
	return prediction