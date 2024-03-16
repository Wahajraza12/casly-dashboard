#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing liberaries 
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM 
import math 
import io
import plotly.express as px
import plotly.graph_objs as go
import urllib
from sklearn.model_selection import train_test_split
import xgboost as xgb
import statsmodels.api as sm
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt # library for plots 
from matplotlib import pyplot #plot for loss of model
from statsmodels import api as sm
from statsmodels.tsa.seasonal import seasonal_decompose 
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import ipywidgets as widgets
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_table
# MSE and MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[ ]:


#Loading data
#importing the data table
tables = pd.read_html("https://www.feanalytics.com/Table.aspx?SavedResultsID=e33df1a2-7b9a-ee11-b204-002248818b97&xl=1&UserID=B2E540A8-D474-4C56-9088-1F155612F7FE&xlRefreshData=1")
#Saving the data into Dateframe
data = tables[0]
#Taking the transpose of table
data = data.T
#Spliting the dates from string format 
#Looping into column and saving the dates in list 
dates = []
for i in range (1,71):
    date = data[1][i][29:39]
    dates.append(date)
#Droping the first and second columns
data = data.drop(columns=[0])
data = data.drop(columns=[1])
#Naming the columns 
data.columns = data.iloc[0]
data = data.iloc[1:].reset_index(drop=True)
#Now droping the Null values rows 
data.dropna(inplace = True)
#Adding the dates 
data['Dates'] = dates
#Converting the date into pandas datetime format
data['Dates'] = pd.to_datetime(data.Dates)
#Setting the Dates as index in our table
data.set_index('Dates', inplace= True)
#Changing the column data into float values
for i in range(len(data.columns)):
    data[data.columns[i]]=data[data.columns[i]].astype(float)


# In[2]:


data = pd.read_csv('data.csv')


# In[3]:


#Converting the date into pandas datetime format
data['Dates'] = pd.to_datetime(data.Dates)
#Setting the Dates as index in our table
data.set_index('Dates', inplace= True)


# In[5]:


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# In[5]:


import warnings
warnings.filterwarnings('ignore')
sarima_data = data[data.columns]
sarima_forecast_data = pd.DataFrame()
sarima_mae = []
sarima_rmse = []
for i in range(len(data.columns)):
    name = data.columns[i]
    model = sm.tsa.statespace.SARIMAX(data[data.columns[i]], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), freq='M')
    results = model.fit(maxiter=100, method='powell')
    sarima_data[name] = results.predict(start=30, end=70, dynamic=True, freq='M')
    sarima_mae.append(mean_absolute_error(data[data.columns[i]][30:70], sarima_data[name][30:70]))
    sarima_rmse.append(root_mean_squared_error(data[data.columns[i]][30:70], sarima_data[name][30:70]))
    from pandas.tseries.offsets import DateOffset
    future_dates = [sarima_data.index[-1] + DateOffset(months=x) for x in range(0, 60)]
    future_datest_df = pd.DataFrame(index=future_dates[1:], columns=sarima_data.columns)
    #Future Forecasting
    Forecasted = results.predict(start=70, end=129, dynamic=True)
    sarima_forecast_data[name] = Forecasted


# In[6]:


#FB Prophet
p_data = data[data.columns]
p_data.reset_index(inplace =True)
fbprophet_mae = []
fbprophet_rmse = []
fbprophet_forecast_data = pd.DataFrame()
for i in range(1,32):
    x =  p_data[[p_data.columns[0] , p_data.columns[i]]]
    x.columns = ['ds','y']
    #Fitting our FB Prophet model 
    fb_model = Prophet()
    fb_model.fit(x)
    future = fb_model.make_future_dataframe(periods=60, freq = 'M')
    prediction=fb_model.predict(future)
    actual_values = x['y'][-len(prediction):]
    fbprophet_mae.append(mean_absolute_error(actual_values, prediction['yhat'][0:70]))
    fbprophet_rmse.append(np.sqrt(mean_squared_error(actual_values, prediction['yhat'][0:70])))

    fig = fb_model.plot(prediction)
    fbprophet_forecast_data[p_data.columns[i]] = prediction['yhat'][-60:]

fbprophet_forecast_data['Dates'] = sarima_forecast_data.index
fbprophet_forecast_data.set_index('Dates', inplace= True)


# In[7]:


#Lstm 
lstm_data = data[data.columns]
lstm_rmse = []
lstm_mae = []
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)
lstm_forecast_data = pd.DataFrame()

# Iterate over each column for forecasting
for i in range(len(data.columns)):
    # Transforming the data using MinMaxScaler
    x = data.iloc[:, i:i+1].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(x)

    # Splitting the dataset into training and testing sets
    training_size = int(len(x) * 0.8)
    test_size = len(x) - training_size
    train_data, test_data = x[0:training_size, :], x[training_size:len(x), :]

    # Reshape into X=t, t+1, t+2, t+3 and Y=t+4
    time_step = 3
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Create the LSTM model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(time_step, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Display model summary
    print('==============================================================================')
    print(data.columns[i])
    print('==============================================================================')
    model.summary()

    # Train the model
    lstm = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=1)

    

    # Perform predictions and evaluate performance metrics
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    look_back= 3
    trainPredictPlot = np.empty_like(x)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(x)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(x)-1, :] = test_predict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(x)) #label = Sectors_legends['Legends'][i])
    plt.plot(trainPredictPlot , label = "Training data")
    plt.plot(testPredictPlot , color = "red" , label = "Testingn data")
    #plt.title(Sectors_legends['Sectors'][i])
    plt.legend(loc='best')
    plt.show()
    
    
    # Concatenate predictions
    whole_predict = np.concatenate((train_predict, test_predict))
    whole_predict = whole_predict.tolist()

    # Store predictions in Lstm_forecast DataFrame
    predict = [item[0] for item in whole_predict]
    lstm_forecast_data[data.columns[i]] = predict[0:60]

    # RMSE
    lstm_rmse1 = math.sqrt(mean_squared_error(y_test, test_predict))


    # MAE
    lstm_mae1 = mean_absolute_error(y_test, test_predict)

    lstm_mae.append(lstm_mae1)
    lstm_rmse.append(lstm_rmse1)
#Setting indexes     
lstm_forecast_data['Dates'] = sarima_forecast_data.index
lstm_forecast_data.set_index('Dates', inplace= True)


# In[8]:


import pandas as pd
import xgboost as xgb

xg_data = data[data.columns]

# Forecast function
def forecast_next_values(model, last_date, num_values=60):
    # Create a DataFrame with the same structure as your original data
    forecast_data = pd.DataFrame(index=pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_values, freq='M'))
    
    # Create features for the forecast data
    X_forecast = create_features(forecast_data)
    
    # Predict using the trained model
    forecast_values = model.predict(X_forecast)
    
    # Add the predicted values to the forecast_data DataFrame
    forecast_data['Prediction'] = forecast_values
    
    return forecast_data

# Create an empty DataFrame to store forecasted values for each column
xgboost_forecast_data = pd.DataFrame()
xgboost_mae = []
xgboost_rmse = []
# Loop through each column in the original dataset
for i in range(data.shape[1]):
    # Extract the column for forecasting
    xg_data = data.iloc[:, i:i+1]  # Select the i-th column

    # Create features function
    def create_features(df):
        df['date'] = df.index
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.weekofyear

        X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
                'dayofyear', 'dayofmonth', 'weekofyear']]

        return X

    # Splitting the dataset into train and test split (70% to training and 30% to testing)
    training_size = int(len(xg_data) * 0.7)
    train_data, test_data = xg_data.iloc[:training_size], xg_data.iloc[training_size:]

    # X and Y of training set
    X_train = create_features(train_data)
    y_train = train_data.iloc[:, 0]  # Assuming only one column in the training set

    # X and Y of testing set
    X_test = create_features(test_data)
    y_test = test_data.iloc[:, 0]  # Assuming only one column in the testing set

    # XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.6, n_estimators=1000)

    # Model fitting on the test data
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              early_stopping_rounds=10,
              verbose=False)
    y_pred = model.predict(X_test)
    xgboost_mae.append(mean_absolute_error(y_test, y_pred))
    
    xgboost_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    # Joining training and test data
    data_all = pd.concat([test_data, train_data], sort=False)

    # Forecast next 60 values
    forecast_df = forecast_next_values(model,data.index[-1], num_values=60)

    # Add forecasted values to the xgboost_forecast DataFrame
    xgboost_forecast_data[data.columns[i]] = forecast_df['Prediction']


# In[12]:


#Rmse and Mae
rmse_mae_result_dataframe = pd.DataFrame()
rmse_mae_result_dataframe["Sectors"] = data.columns
rmse_mae_result_dataframe['Lstm_RMSE'] = lstm_rmse
rmse_mae_result_dataframe['Lstm_MAE'] = lstm_mae
rmse_mae_result_dataframe['Sarima_RMSE'] = sarima_rmse
rmse_mae_result_dataframe['Sarima_MAE'] = sarima_mae
rmse_mae_result_dataframe['Xgboost_RMSE'] = xgboost_rmse
rmse_mae_result_dataframe['Xgboost_MAE'] = xgboost_mae
rmse_mae_result_dataframe['Fbprophet_RMSE'] = fbprophet_rmse
rmse_mae_result_dataframe['Fbprophet_MAE'] = fbprophet_mae


# In[13]:


#Discrete Dataframe function
def calculate_discrete_dataframe(df):
    Discrete = pd.DataFrame({'Discrete Return': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5']})

    for i in range(df.shape[1]):
        temp_list = []
        val = df.iloc[:, i].to_list()

        year1 = np.sum(np.array(val[0:12]))
        temp_list.append(year1)

        year2 = np.sum(np.array(val[12:24]))
        temp_list.append(year2)

        year3 = np.sum(np.array(val[24:36]))
        temp_list.append(year3)

        year4 = np.sum(np.array(val[36:48]))
        temp_list.append(year4)

        year5 = np.sum(np.array(val[48:60]))
        temp_list.append(year5)
        Discrete[df.columns[i]] = temp_list
    
    # Transpose, reset index, and rename columns
    Discrete = Discrete.T.reset_index().rename(columns={'index': 'Discrete Return', 0: 'Year1', 1: 'Year2', 2: 'Year3', 3: 'Year4', 4: 'Year5'})
    Discrete.drop([0], axis=0, inplace=True)
    Discrete = Discrete.reset_index(drop=True)
    
    return Discrete


# In[14]:


#Unit Dataframe
def calculate_unit_dataframe(df):
    Unit = pd.DataFrame({'Unit Price': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5']})

    for i in range(df.shape[1]):
        temp_list = []
        val = df.iloc[:, i].to_list()

        year1 = 1 * (100 + np.sum(np.array(val[0:12])))
        year1 = year1 / 100
        temp_list.append(year1)

        year2 = 1 * (100 + np.sum(np.array(val[12:24])))
        year2 = year2 / 100
        temp_list.append(year2)

        year3 = 1 * (100 + np.sum(np.array(val[24:36])))
        year3 = year3 / 100
        temp_list.append(year3)

        year4 = 1 * (100 + np.sum(np.array(val[36:48])))
        year4 = year4 / 100
        temp_list.append(year4)

        year5 = 1 * (100 + np.sum(np.array(val[48:60])))
        year5 = year5 / 100
        temp_list.append(year5)

        Unit[df.columns[i]] = temp_list
    
        # Transpose, reset index, and rename columns
    Unit = Unit.T.reset_index().rename(columns={'index': 'Unit', 0: 'Year1', 1: 'Year2', 2: 'Year3', 3: 'Year4', 4: 'Year5'})
    Unit.drop([0], axis=0, inplace=True)
    Unit = Unit.reset_index(drop=True)
    return Unit


# In[15]:


#Cumulative Return Function
def calculate_cumulative_returns(discrete_df, unit_df):
    cumulative_returns = discrete_df[[discrete_df.columns[0], discrete_df.columns[1]]]
    cumulative_returns['Year2'] = ((unit_df['Year2']/1) ** (0.5) - 1) * 100
    cumulative_returns['Year3'] = ((unit_df['Year3']/1) ** (0.33) - 1) * 100
    cumulative_returns['Year4'] = ((unit_df['Year4']/1) ** (0.25) - 1) * 100
    cumulative_returns['Year5'] = ((unit_df['Year5']/1) ** (0.20) - 1) * 100
    cumulative_returns.rename(columns={discrete_df.columns[0]: 'Cumulative Returns %'}, inplace=True)
    return cumulative_returns


# In[16]:


def covariance_matrix_table(data):
    # Calculate covariance matrix using NumPy's cov function
    covariance_matrix = np.cov(data, rowvar=False)
    # Convert the covariance matrix to a pandas DataFrame
    columns = [data.columns[i] for i in range(covariance_matrix.shape[0])]
    covariance_df = pd.DataFrame(covariance_matrix, columns=columns, index=columns)
    return covariance_df


# # Gurobipy Optimizer

# In[22]:


import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

def pre_optimize_weights(df, minimum_weight,maximum_weight, total_investment,volitility_value):

    # Define asset names list
    asset_names = ['Alternatives', 'Alternatives', 'Cash and Term Deposits', 'Cash and Term Deposits', 'Fixed Income', 'Commodities', 'International Securities', 'International Securities', 'Australian Securities', 'Australian Securities', 'Australian Securities', 'Australian Securities', 'International Securities', 'International Securities', 'International Securities', 'International Securities', 'International Securities', 'Infrastructure', 'International Securities', 'Private Equity', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Alternatives', 'Property', 'Property', 'Property']
    df = df.astype(float)
    # Define minimum and maximum allocations for each asset class
    min_allocations = {
        'Cash and Term Deposits': 10.0,
        'Fixed Income': 20.0,
        'Australian Securities': 10.0,
        'International Securities': 0.0,
        'Private Equity': 0.0,
        'Property': 0.0,
        'Infrastructure': 0.0,
        'Commodities': 0.0,
        'Alternatives': 0.0
    }
    max_allocations = {
        'Cash and Term Deposits': 60.0,
        'Fixed Income': 60.0,
        'Australian Securities': 50.0,
        'International Securities': 25.0,
        'Private Equity': 20.0,
        'Property': 20.0,
        'Infrastructure': 20.0,
        'Commodities': 10.0,
        'Alternatives': 20.0
    }

    # Store risk values and portfolio returns
    risk_values = []
    portfolio_returns = []

    optimal_model = pd.DataFrame()
    optimal_model['Asset'] = asset_names
    num_total_assets = len(df)

    for year in df.columns:
        # Create a model
        m = gp.Model()

        # Create variables
        weights = m.addVars(num_total_assets, lb=0, ub=20, vtype=gp.GRB.CONTINUOUS, name="weights")
        incidator = m.addVars(num_total_assets, vtype=gp.GRB.BINARY)
        volatility = m.addVar(name="volatility")  # Remove lb=0 from here
        


        for asset in set(asset_names):
            indices = [i for i, name in enumerate(asset_names) if name == asset]
            min_allocation = min_allocations[asset]
            max_allocation = max_allocations[asset]
            m.addConstr(gp.quicksum(weights[i] for i in indices) >= min_allocation, name=f"min_allocation_{asset}")
            m.addConstr(gp.quicksum(weights[i] for i in indices) <= max_allocation, name=f"max_allocation_{asset}")

        m.addConstr(gp.quicksum(weights[i] for i in range(num_total_assets)) == 100, name="total_weights")
        m.addConstrs(weights[i] >= minimum_weight * incidator[i] for i in range(num_total_assets))
        m.addConstrs(weights[i] <= maximum_weight * incidator[i] for i in range(num_total_assets))
        m.addConstr(gp.quicksum(incidator[i] for i in range(num_total_assets)) >= total_investment)

        m.setObjective(gp.quicksum(weights[i] * df[year][i] for i in range(num_total_assets)), GRB.MAXIMIZE)
        cov_matrix = data.cov()
        portfolio_variance = gp.quicksum(weights[i] * weights[j] * cov_matrix.iloc[i, j] for i in range(num_total_assets) for j in range(num_total_assets))
        m.addConstr(volatility * volatility <= volitility_value**2, name="volatility_constraint")
        m.optimize()
        if m.status == GRB.OPTIMAL:
            optimal_weights = np.array([weights[i].x for i in range(num_total_assets)])
            optimal_model[f'Optimal_Weights_{year}'] = optimal_weights
            sumproduct_year = optimal_weights* df[year].values
            optimal_model[f'Sumproduct_{year}'] = sumproduct_year
            
            # Calculate portfolio risk (volatility)
            cov_matrix = (df/100).T.cov()
            portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            risk_values.append(portfolio_risk)

            # Calculate portfolio return
            portfolio_return = np.dot(optimal_weights, df[year].values)
            portfolio_returns.append(portfolio_return)

            # Simulate portfolios for efficient frontier
            num_portfolios = 1000
            simulated_portfolio_returns = []
            simulated_portfolio_risks = []
            for _ in range(num_portfolios):
                random_weights = np.random.random(num_total_assets)
                random_weights /= np.sum(random_weights)
                simulated_portfolio_return = np.dot(random_weights, df[year].values)
                simulated_portfolio_std_dev = np.sqrt(np.dot(random_weights.T, np.dot(cov_matrix, random_weights)))
                simulated_portfolio_returns.append(simulated_portfolio_return)
                simulated_portfolio_risks.append(simulated_portfolio_std_dev)
            # Calculate scaling factors
            risk_scaling_factor = np.max(risk_values) / np.max(simulated_portfolio_risks)
            return_scaling_factor = np.max(portfolio_returns) / np.max(simulated_portfolio_returns)

            # Rescale simulated portfolio risks and returns
            scaled_simulated_portfolio_risks = [risk * risk_scaling_factor for risk in simulated_portfolio_risks]
            scaled_simulated_portfolio_returns = [return_ * return_scaling_factor for return_ in simulated_portfolio_returns]





    return optimal_model, risk_values ,portfolio_returns,scaled_simulated_portfolio_risks,scaled_simulated_portfolio_returns

# Example usage:
# optimal_model, risk_values, portfolio_returns = optimize_weights(df, minimum_weight, total_investment, cov_matrix, maximum_volatility)


# In[21]:


import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

def optimize_weights(df,target_objective, tolerance):

    # Define asset names list
    
    asset_names = ['Alternatives', 'Alternatives', 'Cash and Term Deposits', 'Cash and Term Deposits', 'Fixed Income', 'Commodities', 'International Securities', 'International Securities', 'Australian Securities', 'Australian Securities', 'Australian Securities', 'Australian Securities', 'International Securities', 'International Securities', 'International Securities', 'International Securities', 'International Securities', 'Infrastructure', 'International Securities', 'Private Equity', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Alternatives', 'Property', 'Property', 'Property']
    df = df [df.columns[1:6]]
    df = df.astype(float)
   
    # Define minimum and maximum allocations for each asset class
    min_allocations = {
        'Cash and Term Deposits': 10.0,
        'Fixed Income': 20.0,
        'Australian Securities': 10.0,
        'International Securities': 0.0,
        'Private Equity': 0.0,
        'Property': 0.0,
        'Infrastructure': 0.0,
        'Commodities': 0.0,
        'Alternatives': 0.0
    }
    max_allocations = {
        'Cash and Term Deposits': 60.0,
        'Fixed Income': 60.0,
        'Australian Securities': 50.0,
        'International Securities': 25.0,
        'Private Equity': 20.0,
        'Property': 20.0,
        'Infrastructure': 20.0,
        'Commodities': 10.0,
        'Alternatives': 20.0
    }

    # Store risk values and portfolio returns
    risk_values = []
    portfolio_returns = []

    optimal_model = pd.DataFrame()
    optimal_model['Asset'] = asset_names
    num_total_assets = len(df)

    for year in df.columns:
        # Create a model
        m = gp.Model()

        # Create variables
        weights = m.addVars(num_total_assets, lb=0, ub=20, vtype=gp.GRB.CONTINUOUS, name="weights")
        incidator = m.addVars(num_total_assets, vtype=gp.GRB.BINARY)
        volatility = m.addVar(name="volatility")  # Remove lb=0 from here
        
        # Add constraint to ensure the objective is within the desired range
        m.addConstr(gp.quicksum(weights[i] * df[year][i] for i in range(num_total_assets)) >= target_objective - tolerance,
                    name="objective_lower_bound")
        m.addConstr(gp.quicksum(weights[i] * df[year][i] for i in range(num_total_assets)) <= target_objective + tolerance,
                    name="objective_upper_bound")

        for asset in set(asset_names):
            indices = [i for i, name in enumerate(asset_names) if name == asset]
            min_allocation = min_allocations[asset]
            max_allocation = max_allocations[asset]
            m.addConstr(gp.quicksum(weights[i] for i in indices) >= min_allocation, name=f"min_allocation_{asset}")
            m.addConstr(gp.quicksum(weights[i] for i in indices) <= max_allocation, name=f"max_allocation_{asset}")

        m.addConstr(gp.quicksum(weights[i] for i in range(num_total_assets)) == 100, name="total_weights")

        

        m.setObjective(gp.quicksum(weights[i] * df[year][i] for i in range(num_total_assets)), GRB.MAXIMIZE)

        m.optimize()
        if m.status == GRB.OPTIMAL:
            optimal_weights = np.array([weights[i].x for i in range(num_total_assets)])
            optimal_model[f'Optimal_Weights_{year}'] = optimal_weights
            sumproduct_year = optimal_weights*df[year].values
            optimal_model[f'Sumproduct_{year}'] = sumproduct_year
            
            # Calculate portfolio risk (volatility)
            cov_matrix = (df/100).T.cov()
            portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            risk_values.append(portfolio_risk)

            # Calculate portfolio return
            portfolio = np.dot(optimal_weights, df[year].values)
            portfolio_returns.append(portfolio)

            # Simulate portfolios for efficient frontier
            num_portfolios = 1000
            simulated_portfolio_returns = []
            simulated_portfolio_risks = []
            for _ in range(num_portfolios):
                random_weights = np.random.random(num_total_assets)
                random_weights /= np.sum(random_weights)
                simulated_portfolio_return = np.dot(random_weights, df[year].values)
                simulated_portfolio_std_dev = np.sqrt(np.dot(random_weights.T, np.dot(cov_matrix, random_weights)))
                simulated_portfolio_returns.append(simulated_portfolio_return)
                simulated_portfolio_risks.append(simulated_portfolio_std_dev)
            # Calculate scaling factors
            risk_scaling_factor = np.max(risk_values) / np.max(simulated_portfolio_risks)
            return_scaling_factor = np.max(portfolio_returns) / np.max(simulated_portfolio_returns)

            # Rescale simulated portfolio risks and returns
            scaled_simulated_portfolio_risks = [risk * risk_scaling_factor for risk in simulated_portfolio_risks]
            scaled_simulated_portfolio_returns = [return_ * return_scaling_factor for return_ in simulated_portfolio_returns]





    return optimal_model, risk_values

# Example usage:
# optimal_model, risk_values, portfolio_returns = optimize_weights(df, minimum_weight, total_investment, cov_matrix, maximum_volatility)


# # Dashboard

# In[ ]:


import io
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd


# Importing required functions, assuming they are defined elsewhere
def correlate_table(data):
    
    #Now printing how much each sector has correlating with other sectors.
    null_list = []
    for i in range(31):
        null_list_name  = 'list' + str(i)
        null_list.append(null_list_name)
        null_list[i] = data.corr().values[i]
    index_list=[]
    for i in range(31):
        name  = 'Column' + str(i)
        index_list.append(name)
    for i in range(31):
        index_list[i]  = []
        for x in range(31):

            if null_list[i][x] >= 0.7:
                index_list[i].append(x)
    number_corr = []
    for i in range(31):
        x = (len(index_list[i]))
        number_corr.append(x)
    correlation_table=pd.DataFrame({
        'Sectors':data.columns,
        'Correlation Number': number_corr})
    return correlation_table
def generate_dataframe(dataframe):

    df = calculate_discrete_dataframe(dataframe)
    for i in range(1,6):
        df[f'Categorize_Year{i}'] = "Neutral"
    return df


# Initialize optimal_model as an empty DataFrame
optimal_model = pd.DataFrame()

app = dash.Dash(__name__, external_stylesheets=['assets/all.css'])

asset_names = ['Alternatives', 'Alternatives', 'Cash and Term Deposits', 'Cash and Term Deposits', 'Fixed Income', 'Commodities', 'International Securities', 'International Securities', 'Australian Securities', 'Australian Securities', 'Australian Securities', 'Australian Securities', 'International Securities', 'International Securities', 'International Securities', 'International Securities', 'International Securities', 'Infrastructure', 'International Securities', 'Private Equity', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Fixed Income', 'Alternatives', 'Property', 'Property', 'Property']


historical = calculate_discrete_dataframe(data)
covariance_table = covariance_matrix_table(data)
# Adding 5 new columns with the default value 'Neutral'
lstm_forecast = generate_dataframe(lstm_forecast_data)
sarima_forecast = generate_dataframe(sarima_forecast_data)
xgboost_forecast = generate_dataframe(xgboost_forecast_data)
fbprophet_forecast = generate_dataframe(fbprophet_forecast_data)
df = generate_dataframe(lstm_forecast_data)
df_historical_camulative = calculate_cumulative_returns(calculate_discrete_dataframe(data), calculate_unit_dataframe(data))
df2_lstm = calculate_cumulative_returns(calculate_discrete_dataframe(lstm_forecast_data),
                                   calculate_unit_dataframe(lstm_forecast_data))
df2_sarima = calculate_cumulative_returns(calculate_discrete_dataframe(sarima_forecast_data),
                                     calculate_unit_dataframe(sarima_forecast_data))
df2_xgboost = calculate_cumulative_returns(calculate_discrete_dataframe(xgboost_forecast_data),
                                      calculate_unit_dataframe(xgboost_forecast_data))
df2_fbprophet = calculate_cumulative_returns(calculate_discrete_dataframe(fbprophet_forecast_data),
                                        calculate_unit_dataframe(fbprophet_forecast_data))
df_historical_cumulative_asset = df_historical_camulative
df_historical_cumulative_asset['Asset'] =  asset_names
df_historical_cumulative_asset = df_historical_cumulative_asset.pivot_table(index='Asset', aggfunc='mean')
df_historical_cumulative_asset = df_historical_cumulative_asset.reset_index()
truncated_sector_names = [sector[13:-8] for sector in rmse_mae_result_dataframe['Sectors']]
change_df = sarima_forecast_data
forecast_compare_options = {
    'SARIMA': df2_sarima,
    'XGBoost': df2_xgboost,
    'FBProphet': df2_fbprophet,
    'LSTM':df2_lstm
}
forecast_options = {
    'SARIMA': df2_sarima,
    'XGBoost': df2_xgboost,
    'FBProphet': df2_fbprophet,
    'LSTM':df2_lstm
}

dropdown_options = {
    'LSTM Forecast': lstm_forecast,
    'SARIMA Forecast': sarima_forecast,
    'XGBoost Forecast': xgboost_forecast,
    'FBProphet Forecast': fbprophet_forecast
}
app.layout = html.Div([# Site Header
html.Div([
    html.Div([
        html.A([
            html.Img(src="https://cashelfo.com/wp-content/themes/cashel/img/CFO_symbol_white.png",
                     alt="CashelFamilyOffice Logo", className="logo-icon"),
            html.Img(src="https://cashelfo.com/wp-content/themes/cashel/img/CFO_symbol_white.png",
                     alt="CashelFamilyOffice logo", className="logo-full desktop-only"),
            html.Img(src="https://cashelfo.com/wp-content/themes/cashel/img/CFO_symbol_white.png",
                     alt="CashelFamilyOffice logo", className="logo-full mobile-only"),
        ], href="/", className="logo"),
        html.A("03 9209 9000", href="tel:0392099000", className="get-in-touch"),
        html.A([
            html.Img(src="https://cashelfo.com/wp-content/themes/cashel/img/icon_form_white.png",
                     alt="CashelFamilyOffice form icon", title="Forms", className="header-icon icon"),
            html.Img(src="https://cashelfo.com/wp-content/themes/cashel/img/icon_form_white.png",
                     alt="CashelFamilyOffice form icon", title="Forms", className="header-icon icon-full"),
        ], href="/forms", className=""),
        html.Span([
            html.Img(src="https://cashelfo.com/wp-content/themes/cashel/img/icon_client_white.png",
                     alt="CashelFamilyOffice client icon", title="Client Login", className="header-icon two icon"),
            html.Img(src="https://cashelfo.com/wp-content/themes/cashel/img/icon_client_white.png",
                     alt="CashelFamilyOffice client", title="Client Login", className="header-icon two icon-full"),
            html.Div([
                html.Div([
                    html.Ul([
                        # Uncomment and add list items for client login links if needed
                        # html.Li(html.A("Xplore", href="https://portal.xplorewealthservices.com.au/linearBpms/loginCashelHouse.jsp", target="_blank")),
                        # html.Li(html.A("Preamium", href="https://login.onpraemium.com/CashelFS/", target="_blank")),
                        # html.Li(html.A("Hub24", href="https://my.hub24.com.au/Hub24/Login.aspx", target="_blank")),
                    ])
                ], className="inner")
            ], className="dropdown-content")
        ], className="dropdown"),
        html.Span([
            html.Span("", className=""),
            html.Span("", className="")
        ], className="menu-button"),
    ], className="container")
], id="site-header"),

# Menu Box
html.Div([
    html.Div([
        html.Span([
            html.Span(""),
            html.Span("")
        ], className="menu-button")
    ], className="container"),
    html.Div([
        # Mobile Menu
        html.Div([
            html.Ul([
                html.Li(html.A("Home", href="https://cashelfo.com/")),
                html.Li([
                    html.A("Wealth Services"),
                    html.Ul([
                        html.Li(html.A("Plan", href="https://cashelfo.com/wealth/plan/")),
                        html.Li(html.A("Protect", href="https://cashelfo.com/wealth/protect/")),
                        html.Li(html.A("Invest", href="https://cashelfo.com/wealth/invest/")),
                        html.Li(html.A("Transform", href="https://cashelfo.com/wealth/transform/")),
                        html.Li(html.A("Organise", href="https://cashelfo.com/wealth/organise/")),
                    ], className="sub-menu")
                ]),
                html.Li([
                    html.A("Family Office"),
                    html.Ul([
                        html.Li(html.A("Our Services", href="https://cashelfo.com/our-services/")),
                        html.Li(html.A("Global Product Range", href="https://cashelfo.com/wealth/global-product-range/")),
                        html.Li(html.A("Investment Platforms", href="https://cashelfo.com/wealth/investment-platforms/")),
                        html.Li(html.A("Join Cashel Family", href="https://cashelfo.com/join-cashelfo/")),
                    ], className="sub-menu")
                ]),
                html.Li(html.A("Insights", href="https://cashelfo.com/wealth-insights/")),
                html.Li(html.A("Contact Us", href="https://cashelfo.com/contact-us/")),
                html.Li(html.A("Client Login", href="https://portal.xplorewealthservices.com.au/linearBpms/loginCashelHouse.jsp", target="_blank", rel="noopener")),
            ], id="main-menu", className="navbar-nav")
        ], className="mobile-menu"),

        # Inner
        html.Div([
            # Menu Left
            html.Div([
                html.Ul([
                    html.Li(html.A("Home", href="https://cashelfo.com/")),
                    html.Li(html.A("Wealth Services", href="https://cashelfo.com/wealth/plan/", className="wealth-services")),
                    html.Li(html.A("Family Office", href="https://cashelfo.com/our-services/", className="family-office")),
                    html.Li(html.A("About Us", href="https://cashelfo.com/about-us/", className="about-us")),
                    html.Li(html.A("Insights", href="https://cashelfo.com/wealth-insights/")),
                    html.Li(html.A("Contact Us", href="https://cashelfo.com/contact-us/")),
                ], id="main-menu-desktop", className="navbar-nav")
            ], id="menu-left"),

            # Menu Right
            html.Div([
                html.Ul([
                    html.Li(html.A("Plan", href="https://cashelfo.com/wealth/plan/", id="menu-item-807")),
                    html.Li(html.A("Protect", href="https://cashelfo.com/wealth/protect/", id="menu-item-808")),
                    html.Li(html.A("Invest", href="https://cashelfo.com/wealth/invest/", id="menu-item-804")),
                    html.Li(html.A("Transform", href="https://cashelfo.com/wealth/transform/", id="menu-item-809")),
                    html.Li(html.A("Borrow", href="https://cashelfo.com/wealth/borrow/", id="menu-item-6201")),
                    html.Li(html.A("Organise", href="https://cashelfo.com/wealth/organise/", id="menu-item-806")),
                ], id="wealth-services", className="navbar-nav"),
                html.Ul([
                    html.Li(html.A("About Us", href="https://cashelfo.com/about-us/", id="menu-item-913")),
                    html.Li(html.A("Community", href="https://cashelfo.com/about-us/community/", id="menu-item-911")),
                    html.Li(html.A("Global Partnerships", href="https://cashelfo.com/about-us/global-partnerships/", id="menu-item-912")),
                ], id="about-us", className="navbar-nav"),
                html.Ul([
                    html.Li(html.A("About Us", href="https://cashelfo.com/about-us/")),
                    html.Li(html.A("Community", href="https://cashelfo.com/about-us/community/")),
                    html.Li(html.A("Global Partnerships", href="https://cashelfo.com/about-us/global-partnerships/")),
                ], id="superannuation", className="navbar-nav"),
                html.Ul([
                    html.Li(html.A("Our Services", href="https://cashelfo.com/our-services/", id="menu-item-815")),
                    html.Li(html.A("Global Product Range", href="https://cashelfo.com/wealth/global-product-range/", id="menu-item-817")),
                    html.Li(html.A("Investment Platforms", href="https://cashelfo.com/wealth/investment-platforms/", id="menu-item-818")),
                    html.Li(html.A("Join Cashel Family Office", href="https://cashelfo.com/join-cashelfo/", id="menu-item-5154")),
                ], id="family-office", className="navbar-nav"),
            ], id="menu-right"),
        ], className="inner"),
    ], className="container"),
], id="menu-box"),
    html.H1("Dashboard For Analysis"),
                     # Dropdown to select multiple columns for plotting
                       html.Label("Select Column(s):"),
                       dcc.Dropdown(
                           id='column-dropdown',
                           style = {'padding':'10px'},
                           options=[{'label': col, 'value': col} for col in data.columns],
                           value=[data.columns[0]],  # Default value (can be a list of columns)
                           multi=True
                       ),

                       # Line chart to display the selected column(s) over time
                       dcc.Graph(id='line-chart'),

                       # Dropdown to select multiple forecast methods
                       html.Label("Select Forecast Method(s):"),
                       dcc.Dropdown(
                           id='forecast-method-dropdown',
                           options=[
                               {'label': 'Original Data', 'value': 'original'},
                               {'label': 'LSTM Forecast', 'value': 'lstm'},
                               {'label': 'XGBoost Forecast', 'value': 'xgboost'},
                               {'label': 'SARIMA Forecast', 'value': 'sarima'},
                               {'label': 'FBProphet Forecast', 'value': 'fbprophet'},
                           ],
                           value=['original'],  # Default value (can be a list of methods)
                           multi=True
                       ),

                       # Table to display the data or forecast
                       dash_table.DataTable(
                           id='data-table1',
                           columns=[{'name': col, 'id': col} for col in data.columns],
                           data=data.to_dict('records'),
                           style_table={'height': '400px', 'overflowY': 'auto'},
                           style_cell={'textAlign': 'center', 'fontSize': 12},
                           style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                       ),      html.H4("Evaluation Matric"),
                               dcc.Dropdown(
                                id='metric-dropdown',
                                style = {'padding':'10px'},
                                options=[{'label': col, 'value': col} for col in rmse_mae_result_dataframe.columns[1:]],
                                value='Lstm_RMSE',  # Default value
                                multi=False
                            ),
                            dcc.Graph(id='bar-chart-ev'),
                       
                         html.H1("Customize Dataframe Dashboard"),
                        html.Label("Select Dataframe to Customize:"),
                        dcc.Dropdown(
                            id='dataframe-dropdown',
                            style = {'padding':'10px'},
                            options=[
                                {'label': 'SARIMA DataFrame', 'value': 'sarima_forecast_data'},
                                {'label': 'LSTM DataFrame', 'value': 'lstm_forecast_data'},
                                {'label': 'XGBoost DataFrame', 'value': 'xgboost_forecast_data'},
                                {'label': 'FBProphet DataFrame', 'value': 'fbprophet_forecast_data'}
                            ],
                            value='sarima_forecast_data'
                        ),
                        html.Label("Select Column to Change:"),
                        dcc.Dropdown(
                            id='column-dropdown1',
                            style = {'padding':'10px'},
                            multi=False
                        ),
                        html.Label("Select DataFrame to Change With:"),
                        dcc.Dropdown(
                            id='change-dataframe-dropdown',
                            style = {'padding':'10px'},
                            options=[
                                {'label': 'SARIMA DataFrame', 'value': 'sarima_forecast_data'},
                                {'label': 'LSTM DataFrame', 'value': 'lstm_forecast_data'},
                                {'label': 'XGBoost DataFrame', 'value': 'xgboost_forecast_data'},
                                {'label': 'FBProphet DataFrame', 'value': 'fbprophet_forecast_data'}
                            ],
                            value='lstm_forecast_data'
                        ),
                        html.Button('Change Column', id='change-column-button', className='btn btn-default', n_clicks=0),
                        html.Div(id='dataframe-output'),
                        html.A('Download Modified Data', id='download-link', download="modified_data.csv", href="", target="_blank",
                              style= {
                        'background-color':'white',
                        'color': 'black',
                        'textAlign': 'center',
                        'display': 'inline-block',
                        'lineHeight': '60px',
                        'padding': '0 30px',
                        'textDecoration': 'none',
                        'transition': 'all 0.3s ease-in-out',
                        'outline': 'none !important',
                        'borderRadius': '0px',
                        'fontSize': '13px',
                        'fontWeight': '500',
                        'border': '2px solid light-grey',
                        'minWidth': '220px',
                        'height': '60px',
                        'width': '220px'
                    }),

                       html.H3("Historical Data"),
                       dash_table.DataTable(
                           id='historical_data-table',
                           columns=[{'name': col, 'id': col} for col in historical.columns],
                           data=historical.to_dict('records'),  # Use historical data here
                           style_table={'height': '400px', 'overflowY': 'auto'},
                           style_cell={'textAlign': 'center', 'fontSize': 12},
                           style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                       ),
                       html.H3("Covariance table Data"),
                       dash_table.DataTable(
                           id='covariance_data-table',
                           columns=[{'name': col, 'id': col} for col in covariance_table.columns],
                           data=covariance_table.to_dict('records'),  # Use Covariance data here
                           style_table={'height': '400px', 'overflowY': 'auto'},
                           style_cell={'textAlign': 'center', 'fontSize': 12},
                           style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                       ),
                       
                             dcc.Graph(
                            id='bar-chart',
                            figure={
                                'data': [
                                    {'x': correlate_table(data)['Correlation Number'], 'y': correlate_table(data)['Sectors'], 'type': 'bar', 'orientation': 'h'}
                                ],
                                'layout': {
                                'title': 'Correlation Number by Sectors',
                                'xaxis': {'title': 'Correlation Number', 'automargin': True},
                                'yaxis': {'title': 'Sectors', 'automargin': True, 'tickfont': {'size': 10}},
                                'margin': {'l': 150, 'r': 50, 't': 70, 'b': 70},
                                }
                            }
                        ),
                                        html.Div([         dcc.Dropdown(
                                    id='forecast-compare-dropdown',
                                     style = {'padding':'10px'},
                                    options=[{'label': key, 'value': key} for key in forecast_compare_options.keys()],
                                    value='SARIMA'  # Default value
                                ),
                                html.H4("Historical Cumulative Return"),  # Title for the first table

                                dash_table.DataTable(
                                    id='table1',
                                    columns=[{"name": i, "id": i} for i in df_historical_camulative.columns],
                                    data=df_historical_camulative.to_dict('records'),
                                    style_cell={'font_size': '6.5pt'},  # Adjust font size
                                    style_table={'overflowX': 'scroll', 'maxWidth': '100%'}  # Adjust table size
                                )
                            ], style={'width': '55%', 'display': 'inline-block', 'margin-right': '10px'}),

                            html.Div([
                                html.H4("Forecasted Cumulative Return"),  # Title for the second table

                                html.Div(id='table-container')
                            ], style={'width': '35%', 'display': 'inline-block'}),
                                   html.Div([         dcc.Dropdown(
                    id='forecast-compare-dropdown_asset',
                    style = {'padding':'10px'},
                    options=[{'label': key, 'value': key} for key in forecast_compare_options.keys()],
                    value='SARIMA'  # Default value
                ),
                html.H3("Historical Cumulative Return Asset Wise"),  # Title for the first table

                dash_table.DataTable(
                    id='table1_asset',
                    columns=[{"name": i, "id": i} for i in df_historical_cumulative_asset.columns],
                    data=df_historical_cumulative_asset.to_dict('records'),
                    style_cell={'font_size': '6.5pt'},  # Adjust font size
                    style_table={'overflowX': 'scroll', 'maxWidth': '100%'}  # Adjust table size
                )
            ], style={'width': '55%', 'display': 'inline-block', 'margin-right': '10px'}),

            html.Div([
                html.H3("Forecasted Cumulative Return"),  # Title for the second table

                html.Div(id='table-container_asset')
            ], style={'width': '35%', 'display': 'inline-block'}),
                       html.Hr(style = {'height':'2px','background-color':'black'}),
                       

                       html.H1("HandCraft Method"),

                       dbc.Row([
                           dbc.Col(html.Label("Select Forecast Model:"), width=2),
                           dbc.Col(dcc.Dropdown(
                               id='forecast-model-dropdown',
                               style = {'padding':'10px'},
                               options=[
                                   {'label': 'SARIMA', 'value': 'sarima'},
                                   {'label': 'FB PROPHET', 'value': 'fbprophet'},
                                   {'label': 'LSTM', 'value': 'lstm'},
                                   {'label': 'XGBOOST', 'value': 'xgboost'},
                                   {'label': 'CUSTOMIZE TABLE', 'value': 'customize_table'},
                               ],
                               value='lstm',
                               placeholder='Select Forecast Model'
                           ), width=10),
                       ], className="mb-3"),

                       dbc.Row([
                           dbc.Col(html.Label("Select Column:"), width=2),
                           dbc.Col(dcc.Dropdown(
                               id='column-input',
                               style = {'padding':'10px'},
                               options=[{'label': col, 'value': col} for col in df.columns[1:6]],
                               value='Year1',
                               placeholder='Select Column'
                           ), width=10),
                       ], className="mb-3"),

                       dbc.Row([
                           dbc.Col(html.Label("Row Number:"), width=2),
                           dbc.Col(dcc.Input(
                               id='row-input',
                               className='form-control',
                               style = {'padding':'10px'},
                               type='number',
                               value=0,
                               placeholder='Row Number',
                               min=0,
                               max=len(df) - 1
                           ), width=10),
                       ], className="mb-3"),

                       dbc.Row([
                           dbc.Col(html.Label("Select Multiplier:"), width=2),
                           dbc.Col(dcc.Dropdown(
                               id='multiplier-dropdown',
                               style = {'padding':'10px'},
                               options=[{'label': val, 'value': val} for val in
                                        ['Very strong buy', 'Strong buy', 'Buy', 'Weak buy', 'Neutral', 'Weak sell',
                                         'Sell',
                                         'Strong sell', 'Very strong sell', 'Exclude']],
                               value='Neutral',
                               placeholder='Select Multiplier'
                           ), width=10),
                       ], className="mb-3"),

                       dbc.Row([
                           dbc.Col(html.Label("Input Value (or NaN):"), width=2),
                           dbc.Col(dcc.Input(
                               id='input-value',
                               className='form-control',
                               style = {'padding':'10px'},
                               type='number',
                               value=float('nan'),
                               placeholder='Input Value (or NaN)'
                           ), width=10),
                       ], className="mb-3"),
                       html.Br(),
                       dbc.Row([
                           dbc.Col(html.Button('Update Value', id='update-button', className='btn btn-default'), width=12),
                       ], className="mb-3"),

                       html.Br(),

                       html.Div(id='update-output'),

                       html.Br(),

                       html.H2("Updated DataFrame Values"),

                       dcc.Download(id="download-updated-csv"),  # Add this line

                       dash_table.DataTable(
                           id='data-table',
                           columns=[{'name': col, 'id': col} for col in df.columns],
                           data=df.to_dict('records'),
                           style_table={'overflowX': 'auto', 'height': '400px'},
                           style_cell={'textAlign': 'center', 'fontSize': 12},
                           style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                       ),

                       html.Button('Download Updated CSV', id='download-updated-csv-button', className='btn btn-default'),  # Add this line
                       html.Hr(style = {'height':'2px','background-color':'black'}),
                       html.Br(),

                       html.H2("Optimal Model Values"),

                       dcc.Download(id="download-optimal-csv"),  # Add this line

                       dash_table.DataTable(
                           id='optimal-table',
                           columns=[],  # Initially empty, will be dynamically updated
                           data=[],
                           style_table={'overflowX': 'auto', 'height': '400px'},
                           style_cell={'textAlign': 'center', 'fontSize': 12},
                           style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                       ),
                       html.Button('Download Optimal CSV', id='download-optimal-csv-button', className='btn btn-default'),  # Corrected line

                       html.Br(),

                       dbc.Row([
                           dbc.Col(html.Label("Minimum Weights:"), width=2),
                           dbc.Col(dcc.Input(
                               id='min-weight-input',
                               className='form-control',
                               type='number',
                               value=5,
                               placeholder='Enter Minimum Weights'
                           ), width=10),
                       ], className="mb-3"),
                        html.Br(), 
                           dbc.Row([
                           dbc.Col(html.Label("Maximum Weights:"), width=2),
                           dbc.Col(dcc.Input(
                               id='max-weight-input',
                               className='form-control',
                               type='number',
                               value=15,
                               placeholder='Enter Maximum Weights'
                           ), width=10),
                       ], className="mb-3"),
                        html.Br(),
                       dbc.Row([
                           dbc.Col(html.Label("Total Investments:"), width=2),
                           dbc.Col(dcc.Input(
                               id='total-investment-input',
                               className='form-control',
                               type='number',
                               value=12,
                               placeholder='Enter Total Investments'
                           ), width=10),
                       ], className="mb-3"),
                       html.Br(),
                           dbc.Row([
                            dbc.Col(html.Label("Maximum Volatility:"), width=2),
                            dbc.Col(dcc.Input(
                                id='volitility-input',
                                className='form-control',
                                type='number',
                                value=13.26,
                                placeholder='Enter Maximum Volatility'
                            ), width=10),
                        ], className="mb-3"),
                        html.Br(),
                       dbc.Row([
                           dbc.Col(html.Button('Optimize', id='optimize-button', className='btn btn-default'), width=12),
                           
                       ], className="mb-3") 
                       ,    dcc.Store(id='user-modified-data', data={}),
                        dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
                       html.H4("Distributions of Weights"),
                       dash_table.DataTable(
                           id='total-table',
                           columns=[],  # Initially empty, will be dynamically updated
                           data=[],
                           style_table={'overflowX': 'auto', 'height': '400px'},
                           style_cell={'textAlign': 'center', 'fontSize': 12},
                           style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                       ),
                        
                   html.Br(),
                html.H2("Model Optimize Statics Data"),
                html.Div(className= 'Model_optimize',children=[
                    dcc.Graph(
                        id='discrete-period-return',
                        figure={
                            'data': [
                                {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Data 1'},
                                # Add more traces with names as needed
                            ],
                            'layout': {
                                'title': 'Discrete Period Return',
                                'legend': {'x': 0, 'y': 1},
                            }
                        }
                    ),
                    dcc.Graph(
                        id='cumulative-return',
                        figure={
                            'data': [
                                {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Data 2'},
                                # Add more traces with names as needed
                            ],
                            'layout': {
                                'title': 'Cumulative Return',
                                'legend': {'x': 0, 'y': 1},
                            }
                        }
                    ),
                    dcc.Graph(
                        id='portfolio-value',
                        figure={
                            'data': [
                                {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Data 3'},
                                # Add more traces with names as needed
                            ],
                            'layout': {
                                'title': 'Portfolio Value',
                                'legend': {'x': 0, 'y': 1},
                            }
                        }
                    ),
                    dcc.Graph(
                        id='risk',
                        figure={
                            'data': [
                                {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Data 4'},
                                # Add more traces with names as needed
                            ],
                            'layout': {
                                'title': 'Risk',
                                'legend': {'x': 0, 'y': 1},
                            }
                        }
                    ),
                ]),
                html.Br(),

                html.H2("Historical Optimize Statics Data"),
                html.Div(className= 'Historical_optimize',children=[
                    dcc.Graph(
                        id='Hdiscrete-period-return',
                        figure={
                            'data': [
                                {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Historical Data 1'},
                                # Add more traces with names as needed
                            ],
                            'layout': {
                                'title': 'Historical Discrete Period Return',
                                'legend': {'x': 0, 'y': 1},
                            }
                        }
                    ),
                    dcc.Graph(
                        id='Hcumulative-return',
                        figure={
                            'data': [
                                {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Historical Data 2'},
                                # Add more traces with names as needed
                            ],
                            'layout': {
                                'title': 'Historical Cumulative Return',
                                'legend': {'x': 0, 'y': 1},
                            }
                        }
                    ),
                    dcc.Graph(
                        id='Hportfolio-value',
                        figure={
                            'data': [
                                {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Historical Data 3'},
                                # Add more traces with names as needed
                            ],
                            'layout': {
                                'title': 'Historical Portfolio Value',
                                'legend': {'x': 0, 'y': 1},
                            }
                        }
                    ),
                    dcc.Graph(
                        id='historical_risk',
                        figure={
                            'data': [
                                {'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': [], 'name': 'Historical Data 4'},
                                # Add more traces with names as needed
                            ],
                            'layout': {
                                'title': 'Historical Risk',
                                'legend': {'x': 0, 'y': 1},
                            }
                        }
                    )
                ]),html.Div(dcc.Graph(id='efficient-frontier-plot')), 
                    html.Div([
                        html.H3("Set Objective "),
                        html.Label('Select DataFrame:'),
                        dcc.Dropdown(
                            id='set_objective_dropdown',
                            style={'padding':'2px'},
                            options=[{'label': key, 'value': key} for key in dropdown_options.keys()],
                            value='LSTM Forecast'
                        ),
                        html.Label('Set Objective:'),html.Br(),
                        dcc.Input(id='set_objective', type='number', value=0,className='form-control'),
                        html.Br(),
                        html.Label('Set Tolerance:'), html.Br(),
                        dcc.Input(id='set_tolerance', type='number', value=0,className='form-control'),
                        html.Br(),
                        html.Button('Optimize', id='set_optimize_button',className='btn btn-default', n_clicks=0),
                        dcc.Graph(id='set_objective_line-chart'),
                        dash_table.DataTable(id='set_objective_table',                           
                            style_table={'overflowX': 'auto', 'height': '400px'},
                           style_cell={'textAlign': 'center', 'fontSize': 12},
                           style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}),
                        
                    ]),
    html.Footer(style = {'background-color':'whilte'},children=[
    html.Div([
        html.Div([
            html.Img(src="https://cashelfo.com/wp-content/themes/cashel/img/CFO_logo_white.png", alt="CFO Logo")
        ], className="col-lg-6 col-sm-12"),

        html.Div([
            html.H5([
                html.A('[email protected]', href="mailto: [email protected]", className="_cf_email_", 
                       **{'data-cfemail': 'd1bbbeb8bf91b2b0a2b9b4bdb7beffb2bebc'}),
                html.A("03 9209 9000", href="tel:0392099000")
            ])
        ], className="col-lg-6 col-sm-12")
    ], className="container top-container"),

    # Navigation Links
    html.Div([
        html.Div([
            html.H4("Wealth Services"),
            html.Ul([
                html.Li(html.A("Plan", href="/wealth/plan/")),
                html.Li(html.A("Protect", href="/wealth/protect/")),
                html.Li(html.A("Invest", href="/wealth/invest/")),
                html.Li(html.A("Borrow", href="/wealth/borrow/")),
                html.Li(html.A("Transform", href="/wealth/transform/")),
                html.Li(html.A("Organise", href="/wealth/organise/")),
            ])
        ], id="text-2", className="widget widget_text"),

        html.Div([
            html.H4("Family Office"),
            html.Ul([
                html.Li(html.A("Our Services", href="/our-services/")),
                html.Li(html.A("Global Product Range", href="/wealth/global-product-range/")),
                html.Li(html.A("Investment Platforms", href="/wealth/investment-platforms/")),
                html.Li(html.A("Join Cashel Family Office", href="/join-cashelfo/")),
            ])
        ], id="text-4", className="widget widget_text"),

        html.Div([
            html.H4("About Us"),
            html.Ul([
                html.Li(html.A("About Us", href="/about-us/about-us/")),
                html.Li(html.A("Global Partnerships", href="/about-us/global-partnerships/")),
                html.Li(html.A("Community", href="/about-us/community/")),
            ])
        ], id="text-5", className="widget widget_text"),

        html.Div([
            html.H4("Useful links"),
            html.Ul([
                html.Li(html.A("Privacy Policy", href="/privacy-policy/")),
                html.Li(html.A("Conflicts of Interest policy", href="/conflicts-of-interest-policy/")),
                html.Li(html.A("Forms", href="/forms")),
            ])
        ], id="text-6", className="widget widget_text"),

        html.Div([
            html.Div([
                html.A(html.Img(src="https://cashelfo.com/wp-content/uploads/2023/12/apple_app_store.png", alt=""),
                       href="https://apps.apple.com/au/app/cashel-family-office/id6473882152", target="_blank",
                       rel="noopener", style={"max-width": "128px !important", "max-height": "min-content"}),
                html.Br(),
                html.A(html.Img(src="https://cashelfo.com/wp-content/uploads/2023/12/google_play_badge.png", alt=""),
                       href="https://play.google.com/store/apps/details?id=com.softobiz.casher_family_office&hl=en_US&gl=US",
                       target="_blank", rel="noopener", style={"max-width": "128px !important", "max-height": "min-content"}),
            ], className="textwidget active")
        ], className="widget widget_text", style={"padding": "0px", "padding-top": "0px"})
    ], className="container", style={"display": "flex", "justify-content": "space-around"})
                ]),
                        ])

# Callback to update the bar chart based on user selection
@app.callback(
    Output('bar-chart-ev', 'figure'),
    [Input('metric-dropdown', 'value')]
)
def update_chart(selected_metric):
    # Prepare data for plotting
    bar_data = []
    for sector, truncated_sector in zip(rmse_mae_result_dataframe['Sectors'], truncated_sector_names):
        bar_data.append(go.Bar(
            x=[truncated_sector],
            y=[rmse_mae_result_dataframe.loc[rmse_mae_result_dataframe['Sectors'] == sector, selected_metric].iloc[0]],
            name=f'{truncated_sector}'
        ))

    # Define layout
    layout = go.Layout(
        title=f'Bar Chart of {selected_metric} for Each Sector',
        xaxis={'title': 'Sectors'},
        yaxis={'title': selected_metric}
    )

    # Return the figure
    return {'data': bar_data, 'layout': layout}

# Define callback to update column dropdown based on selected dataframe
@app.callback(
    Output('column-dropdown1', 'options'),
    [Input('dataframe-dropdown', 'value')]
)
def update_column_dropdown(selected_dataframe):
    dataframe = globals()[selected_dataframe]  # Get the selected dataframe
    options = [{'label': col, 'value': col} for col in dataframe.columns]
    return options

# Define callback to display customized dataframe
@app.callback(
    Output('dataframe-output', 'children'),
    [Input('change-column-button', 'n_clicks')],
    [State('dataframe-dropdown', 'value'),
     State('column-dropdown1', 'value'),
     State('change-dataframe-dropdown', 'value')]
)
def change_column(n_clicks, selected_dataframe, column, change_dataframe):
    if n_clicks > 0 and column is not None:
        global change_df 
        change_df = pd.DataFrame(globals()[selected_dataframe])
        change_df[column] = globals()[change_dataframe][column]
        return dash_table.DataTable(
            id='table',
            columns=[{'name': col, 'id': col} for col in change_df.columns],
            data=change_df.to_dict('records'),
            style_table={'height': '400px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'center', 'fontSize': 12},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
        )

# Define callback to download modified data
@app.callback(
    Output('download-link', 'href'),
    [Input('change-column-button', 'n_clicks')],
    [State('dataframe-dropdown', 'value')]
)
def download_modified_data(n_clicks, selected_dataframe):
    if n_clicks > 0:
        change_df = pd.DataFrame(globals()[selected_dataframe])
        csv_string = change_df.to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
        return csv_string
    else:
        return ""

# Define callback to update the line chart and table based on the selected column and forecast method
@app.callback(
    [Output('line-chart', 'figure'),
     Output('data-table1', 'data')],
    [Input('column-dropdown', 'value'),
     Input('forecast-method-dropdown', 'value')]
)
def update_chart_and_table(selected_columns, selected_forecast_methods):
    # Create an empty figure
    fig = px.line(title="Selected Columns over Time")

    # Define a list of colors for the lines
    line_colors = px.colors.qualitative.Set1

    for idx, selected_column in enumerate(selected_columns):
        for forecast_method in selected_forecast_methods:
            line_color = line_colors[idx % len(line_colors)]  # Cycle through colors for each selected column

            if forecast_method == 'original':
                # Plot the original data
                fig.add_trace(px.line(data, x=data.index, y=selected_column, title=f'{selected_column} over time').update_traces(line=dict(color=line_color)).data[0])
            elif forecast_method == 'lstm':
                # Assuming you have 'lstm_forecast' DataFrame
                forecast_data = lstm_forecast_data[selected_column]

                fig.add_trace(px.line(lstm_forecast_data, x=lstm_forecast_data.index, y=selected_column, title=f'LSTM Forecast: {selected_column} (RMSE: {2.3:.2f}, MAE: {1.4:.2f})').update_traces(line=dict(color=line_color)).data[0])
            elif forecast_method == 'xgboost':
                # Assuming you have 'xgboost_forecast' DataFrame
                forecast_data = xgboost_forecast_data[selected_column]

                fig.add_trace(px.line(xgboost_forecast_data, x=xgboost_forecast_data.index, y=selected_column, title=f'XGBoost Forecast: {selected_column} (RMSE: {4:.2f}, MAE: {2:.2f})').update_traces(line=dict(color=line_color)).data[0])
            elif forecast_method == 'sarima':
                # Assuming you have 'sarima_forecast' DataFrame
                forecast_data = sarima_forecast_data[selected_column]

                fig.add_trace(px.line(sarima_forecast_data, x=sarima_forecast_data.index, y=selected_column, title=f'SARIMA Forecast: {selected_column} (RMSE: {6:.2f}, MAE: {1:.2f})').update_traces(line=dict(color=line_color)).data[0])
            elif forecast_method == 'fbprophet':
                # Assuming you have 'fbprophet_forecast' DataFrame
                forecast_data = fbprophet_forecast_data[selected_column]

                fig.add_trace(px.line(fbprophet_forecast_data, x=fbprophet_forecast_data.index, y=selected_column, title=f'FBProphet Forecast: {selected_column} (RMSE: {7:.2f}, MAE: {4.53:.2f})').update_traces(line=dict(color=line_color)).data[0])

    return fig, data.to_dict('records')
# Define callback to update forecast table based on dropdown selection
@app.callback(
    Output('table-container', 'children'),
    [Input('forecast-compare-dropdown', 'value')]
)
def update_forecast_table(selected_option):
    selected_df = forecast_options[selected_option]
    return dash_table.DataTable(
        id='table2',
        columns=[{"name": i, "id": i} for i in selected_df.columns[1:]],
        data=selected_df.to_dict('records'),
        style_cell={'font_size': '6.5pt'},  # Adjust font size
        style_table={'overflowX': 'scroll', 'maxWidth': '100%'}  # Adjust table size
    )
# Define callback to update forecast table based on dropdown selection
@app.callback(
    Output('table-container_asset', 'children'),
    [Input('forecast-compare-dropdown_asset', 'value')]
)
def update_forecast_table_asset(selected_option):
    selected_df = forecast_options[selected_option]
    selected_df["Asset"] = asset_names
    selected_df = selected_df.pivot_table(index='Asset', aggfunc='mean')
    selected_df = selected_df.reset_index()
    return dash_table.DataTable(
        id='table2',
        columns=[{"name": i, "id": i} for i in selected_df.columns[1:]],
        
        data=selected_df.to_dict('records'),
        style_cell={'font_size': '6.5pt'},  # Adjust font size
        style_table={'overflowX': 'scroll', 'maxWidth': '100%'}  # Adjust table size
    )


# Function to handle "Update Value" button click
@app.callback(
    [Output('update-output', 'children'),
     Output('data-table', 'columns'),
     Output('data-table', 'data')],
    [Input('update-button', 'n_clicks')],
    [State('forecast-model-dropdown', 'value'),
     State('column-input', 'value'),
     State('row-input', 'value'),
     State('multiplier-dropdown', 'value'),
     State('input-value', 'value')]
)
def update_value(n_clicks, selected_forecast_model, selected_column, selected_row, multiplier_str, input_value):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    global change_df
    customize_table = generate_dataframe(change_df)
    # Get the selected forecast model dataframe
    if selected_forecast_model == 'sarima':
        df = sarima_forecast
    elif selected_forecast_model == 'xgboost':
        df = xgboost_forecast
    elif selected_forecast_model == 'fbprophet':
        df = fbprophet_forecast
    elif selected_forecast_model == 'customize table':
        df = customize_table
    else:  # Default to LSTM
        df = lstm_forecast

    # Update the dataframe based on the user input
    if pd.isna(input_value):
        # If input_value is NaN, update using multiplier
        updated_value = df.at[selected_row, selected_column] * get_multiplier_value(multiplier_str)
    else:
        # If input_value is provided, use it
        updated_value = input_value

    # Update the dataframe
    df.at[selected_row, selected_column] = updated_value

    # Update the 'Categorize' column
    categorize_column = f'Categorize_{selected_column}'
    df.at[selected_row, categorize_column] = multiplier_str

    # Prepare data for DataTable
    columns = [{'name': col, 'id': col} for col in df.columns]
    data = df.to_dict('records')

    return f"Successfully Updated On {selected_column} row {selected_row}", columns, data


def get_multiplier_value(multiplier_str):
    multiplier_mapping = {
        'Very strong buy': 1.2,
        'Strong buy': 1.15,
        'Buy': 1.1,
        'Weak buy': 1.05,
        'Neutral': 1,
        'Weak sell': 0.95,
        'Sell': 0.9,
        'Strong sell': 0.85,
        'Very strong sell': 0.8,
        'Exclude': 0,
    }
    return multiplier_mapping.get(multiplier_str, 1)


# Callback to download the updated CSV
@app.callback(
    Output('download-updated-csv', 'data'),
    [Input('download-updated-csv-button', 'n_clicks')],
    [State('forecast-model-dropdown', 'value')]
)
def download_updated_csv(n_clicks, selected_forecast_model):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    buffer = io.StringIO()
    if selected_forecast_model == 'sarima':
        sarima_forecast.to_csv(buffer, index=False)
    elif selected_forecast_model == 'xgboost':
        xgboost_forecast.to_csv(buffer, index=False)
    elif selected_forecast_model == 'fbprophet':
        fbprophet_forecast.to_csv(buffer, index=False)
    elif selected_forecast_model == 'customize table':
        customize_table.to_csv(buffer,index=False)
    else:  # Default to LSTM
        lstm_forecast.to_csv(buffer, index=False)
    buffer.seek(0)

    return dict(content=buffer.getvalue(), filename='updated_dataframe.csv')


# Callback to download the optimal CSV
@app.callback(
    Output('download-optimal-csv', 'data'),
    [Input('download-optimal-csv-button', 'n_clicks')],
    prevent_initial_call=True
)
def download_optimal_csv(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    buffer = io.StringIO()
    optimal_model.to_csv(buffer, index=False)  # Replace df_optimal with your optimal DataFrame
    buffer.seek(0)

    return dict(content=buffer.getvalue(), filename='optimal_model.csv')
@app.callback(
    [Output('set_objective_line-chart', 'figure'),
     Output('set_objective_table', 'data')],
    [Input('set_optimize_button', 'n_clicks')],
    [dash.dependencies.State('set_objective_dropdown', 'value'),
     dash.dependencies.State('set_objective', 'value'),
     dash.dependencies.State('set_tolerance', 'value')]
)
def update_output(n_clicks, selected_df, objective, tolerance):
    if n_clicks > 0:
        # Assuming you have the optimize_weights function defined above
        df = dropdown_options[selected_df]
        
        df.index = pd.date_range(start=pd.Timestamp.now().date(), periods=len(df), freq='D')
        set_optimal_model, set_risk_values = optimize_weights(df, objective, tolerance)

        # Create line chart
        line_chart = go.Figure()
        line_chart.add_trace(go.Scatter(x=['Year1','Year2','Year3','Year4','Year5'], y=set_risk_values, mode='lines', name='Risk Values'))
        line_chart.update_layout(title='Risk Values Over Time')

        # Prepare data for table
        table_data = set_optimal_model.to_dict('records')

        return line_chart, table_data
    else:
        return {}, [] 


# Function to handle "Optimize" button click
@app.callback(
    [Output('optimal-table', 'columns'),
     Output('optimal-table', 'data'),
     Output('total-table', 'columns'),
     Output('total-table', 'data'),
     Output('discrete-period-return', 'figure'),
     Output('cumulative-return', 'figure'),
     Output('portfolio-value', 'figure'),
     Output('Hdiscrete-period-return', 'figure'),
     Output('Hcumulative-return', 'figure'),
     Output('Hportfolio-value', 'figure'),
     Output('risk', 'figure'),
     Output('historical_risk', 'figure'),
     Output('efficient-frontier-plot', 'figure')],
    [Input('optimize-button', 'n_clicks')],
    [State('data-table', 'data'),
     State('min-weight-input', 'value'),
     State('max-weight-input', 'value'),
     State('total-investment-input', 'value'),
     State('volitility-input', 'value')],
    
    
    prevent_initial_call=True
)
def optimize_button_click(n_clicks, data_table_data, min_weight,max_weight, total_investment, volitility):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    global optimal_model, historical_optimize
    
    # Extract updated data from data_table_data
    updated_df = pd.DataFrame(data_table_data)
    updated_df = updated_df[updated_df.columns[1:6]]  # Assuming the relevant columns are from index 1 to 5
    
    # Optimize the weights using the updated data
    optimal_model, risk,portfolio_returns,simulated_portfolio_risks,simulated_portfolio_return = pre_optimize_weights(updated_df , minimum_weight=min_weight, maximum_weight=max_weight,total_investment=total_investment,
                                           volitility_value=volitility)
    historical_optimize_model , historical_risk,p,sr,spr = pre_optimize_weights(historical[historical.columns[1:6]], minimum_weight=min_weight,maximum_weight=max_weight,
    total_investment=total_investment, volitility_value=volitility)
    # Process and calculate figures
    optimal_columns = [{'name': col, 'id': col} for col in optimal_model.columns]
    
    discreate_period_return = [optimal_model['Sumproduct_Year1'].sum() / 100,
                               optimal_model['Sumproduct_Year2'].sum() / 100,
                               optimal_model['Sumproduct_Year3'].sum() / 100,
                               optimal_model['Sumproduct_Year4'].sum() / 100,
                               optimal_model['Sumproduct_Year5'].sum() / 100]

    # Optimize table
    cumulative_return = []
    v = 100
    for i in range(5):
        v = v + discreate_period_return[i]
        cumulative_return.append(v)
    portfolio_value = []
    v = 1000000
    for i in range(5):
        v = (v + (discreate_period_return[i] * 100) * 100)
        portfolio_value.append(v)
    
    # Historical return
    Hdiscreate_period_return = [historical_optimize_model['Sumproduct_Year1'].sum() / 100,
                                historical_optimize_model['Sumproduct_Year2'].sum() / 100,
                                historical_optimize_model['Sumproduct_Year3'].sum() / 100,
                                historical_optimize_model['Sumproduct_Year4'].sum() / 100,
                                historical_optimize_model['Sumproduct_Year5'].sum() / 100]
    Hcumulative_return = []
    v = 100
    for i in range(5):
        v = v + Hdiscreate_period_return[i]
        Hcumulative_return.append(v)
    Hportfolio_value = []
    v = 1000000
    for i in range(5):
        v = (v + (Hdiscreate_period_return[i] * 100) * 100)
        Hportfolio_value.append(v)
       # Update the optimize model figures
    discrete_fig = {
        'data': [{'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': discreate_period_return, 'type': 'bar', 'name': 'Discrete Period Return'}],
        'layout': {'title': 'Discrete Return', 'height': 400, 'width': 400}
    }

    cumulative_fig = {
        'data': [{'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': cumulative_return, 'type': 'line', 'name': 'Cumulative Return'}],
        'layout': {'title': 'Cumulative Return', 'height': 400, 'width': 400}
    }

    portfolio_fig = {
        'data': [{'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': portfolio_value, 'type': 'line', 'name': 'Portfolio Value'}],
        'layout': {'title': 'Portfolio Value', 'height': 400, 'width': 400}
    }

    # Update the historical model figures
    Hdiscrete_fig = {
        'data': [{'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': Hdiscreate_period_return, 'type': 'bar', 'name': 'Historical Discrete Period Return'}],
        'layout': {'title': 'Historical Discrete Return', 'height': 400, 'width': 400}
    }

    Hcumulative_fig = {
        'data': [{'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': Hcumulative_return, 'type': 'line', 'name': 'Historical Cumulative Return'}],
        'layout': {'title': 'Historical Cumulative Return', 'height': 400, 'width': 400}
    }

    Hportfolio_fig = {
        'data': [{'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': Hportfolio_value, 'type': 'line', 'name': 'Historical Portfolio Value'}],
        'layout': {'title': 'Historical Portfolio Value', 'height': 400, 'width': 400}
    }

    forecast_volatility = {
        'data': [{'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': risk, 'type': 'line', 'name': 'Forecast Volatility'}],
        'layout': {'title': 'Forecast Volatility', 'height': 400, 'width': 400}
    }

    historical_volatility = {
        'data': [{'x': ['Year1', 'Year2', 'Year3', 'Year4', 'Year5'], 'y': historical_risk, 'type': 'line', 'name': 'Historical Volatility'}],
        'layout': {'title': 'Historical Volatility', 'height': 400, 'width': 400}
    }


    total_table = optimal_model.pivot_table(index='Asset', aggfunc='sum')
    total_table.reset_index(inplace =True)
    total_table_columns = [{'name': col, 'id': col} for col in total_table.columns[0:6]]
    trace_simulated = go.Scatter(
        x=simulated_portfolio_risks,
        y=simulated_portfolio_return,
        mode='markers',
        marker=dict(color='blue', opacity=0.2),
        name='Simulated Portfolios'
    )
    
    # Scatter plot trace for optimal portfolios
    trace_optimal = go.Scatter(
        x=risk,
        y=portfolio_returns,
        mode='markers',
        marker=dict(color='red', symbol='star'),
        name='Optimal Portfolios'
    )
    
    layout = go.Layout(
        title='Efficient Frontier',
        xaxis=dict(title='Volatility'),
        yaxis=dict(title='Return'),
        legend=dict(x=0, y=1)
    )
    
    
    return optimal_columns, optimal_model.to_dict('records'), total_table_columns, total_table.to_dict('records'), discrete_fig, cumulative_fig, portfolio_fig, Hdiscrete_fig, Hcumulative_fig, Hportfolio_fig, forecast_volatility, historical_volatility, {'data': [trace_simulated, trace_optimal], 'layout': layout}



if __name__ == '__main__':
    app.run_server(debug=True)

