"""
The purpose of this script is to take production part data from the factories 
 and subtract surplus part data to attain a "goal" for optimal part production
 for a given part. It then takes car sales data to build a model that predicts
 what the optimal weekly production of a specific part should be given the sales
 of the three different models the company produces. Note this code makes one 
 model per part number.
"""
import utils
import pandas as pd
import xgboost as xgb
from sklearn.model_selection  import train_test_split
import matplotlib.pyplot as plt
import sys
import pickle
import cdsw

#This handles inputs in case you'd like to pass it the forecasting for next week's models
# or the part that the model will be created for. It defaults to our "experimental engine"
if (len(sys.argv) == 4):
  model_C_sales = sys.argv[1].upper()
  model_D_sales = sys.argv[2].upper()
  model_R_sales = sys.argv[3].upper()
  part_no = sys.argv[4].upper()
elif (len(sys.argv) != 3):
  model_C_sales = 42
  model_D_sales = 55
  model_R_sales = 30
  part_no = 'a42CLDR'

#put together our next week's forecast so we can predict the part quantity needed later
sales_forecast = {'Model C': [model_C_sales],'Model D': [model_D_sales], 'Model R': [model_R_sales]}
df_forecast = pd.DataFrame(sales_forecast)

#This collects all of our data and does the legwork of combining datasets to get us
#  the training data in the format we want
print('Collecting last year worth of data to build model')
final_df = utils.collect_format_data(part_no)

#Split into test/train sets
X = final_df.drop(columns = "goal_parts", axis=1, inplace=False)
y = final_df.goal_parts.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
data_dmatrix = xgb.DMatrix(data=X,label=y)

#Create a gradient boost regression model 
params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=50)

#Do some cross-validation to make sure model is not terrible
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=10,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

#Plot the importance of each feature
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()

#Test the final error
final_rmse = cv_results["test-rmse-mean"].iat[-1]
print("Root Mean Std. Err : ", final_rmse)

#Show predictions for next week using forecasted car production
data_newforecast = xgb.DMatrix(data=df_forecast)
new_preds = xg_reg.predict(data_newforecast)
print("Predicted weekly production for Part No ", part_no, ": ", new_preds[0])

#Save model as pickle file
picklefile = part_no + '.pickle'
pickle.dump(xg_reg, open(picklefile, 'wb'))

cdsw.track_metric('RSME', final_rmse)
cdsw.track_metric('Estimated Part Production', new_preds[0])

"""
def predictParts():
    loaded_model = pickle.load(open(picklefile, 'rb'))
    result = loaded_model.predict(data_newforecast)
    print(result)
    return result
    """
