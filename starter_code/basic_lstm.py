
# 1
import os
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 2
from sklearn.preprocessing import MinMaxScaler
# 3
import keras.backend
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


# 1
# First method: APi token in header
iex_key = os.environ.get('IEX_KEY')

headers = {
    'Authorization':'Token ' + iex_key
}

payload = {

}

with requests.get(url, headers=headers, params=payload) as r:
    data1 = r.json()
    data2 = json.loads(r.text)

headers = { 'Authorization' : 'Token ' + token }
r = requests.get('https://address.of.opengear/api/v1/serialPorts/', headers=headers, verify=False)
j = json.loads(r.text)
print json.dumps(j, indent=4)
# /1

data = pd.read_csv('data/AAPL.csv')
data_train, data_holdout = data.iloc[200:, :], data.iloc[:200, :]

data_train[['Open', 'Adj Close']].plot()
data_holdout[['Open', 'Adj Close']].plot()

array_train = data_train.iloc[:, -1].values
array_holdout = data_holdout.iloc[:, -1].values

# 2
# His data prep
scaler = MinMaxScaler(feature_range = (0, 1))
array_train_scaled = scaler.fit_transform(array_train.reshape(-1, 1))
array_train_scaled
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 3826):
    X_train.append(array_train_scaled[i-60:i, 0])
    y_train.append(array_train_scaled[i, 0])
print('----', X_train[-10])
print('----', y_train[-10])
X_train, y_train = np.array(X_train), np.array(y_train)
print('----', X_train[-10])
print('----', y_train[-10])
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print('----', X_train[-10])

# /2

# >>> Unimplemented

from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
print(tscv)

for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# >>> /Unimplemented

# In[57]:


print(X_train.shape, '---', X_debug.shape)


# In[55]:


y_debug = y_train[0:3]
y_debug


# In[48]:


# Params
lstm_params_in = {
    'units':50,
    'return_sequences':True,
    'input_shape':(X_debug.shape[1], 1)
    #'input_shape':(X_train.shape[1], 1)
}
lstm_params_body = {
    'units':50,
    'return_sequences':True
}
lstm_params_end = {
    'units':50
}
output_params = {
    'units':1
}

compile_params = {
    'optimizer':'adam',
    'loss':'mean_squared_error'
}

# Init
model = Sequential()
# Input
model.add(LSTM(**lstm_params_in))
model.add(Dropout(0.2))
model.add(LSTM(**lstm_params_body))
model.add(Dropout(0.2))
model.add(LSTM(**lstm_params_end))
model.add(Dropout(0.2))
# Output
model.add(Dense(**output_params))

# Compile
model.compile(**compile_params)


# In[88]:


model.summary()


# In[89]:


fit_params = {
    'epochs':100,
    'batch_size':32
}

#model.fit(X_debug, y_debug, **fit_params)
model.fit(X_train, y_train, **fit_params)


# ### Visualization ALL

# In[61]:


# SVG Graph
from keras.utils

plot_model(
    model,
    to_file='model.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB'
)
#SVG(model_to_dot(model).create(prog='dot', format='svg'))


# In[69]:


# History object
#history = model.fit(X_debug, y_debug, **fit_params, verbose = True)


# In[91]:


# LSTM 1
model_json = model.to_json()
with open('resources/lstm_1.json', 'w') as json_file:
    json_file.write(model_json)
    model.save_weights('lstm_1.h5')


# In[71]:


# Plot training loss
plt.plot(history.history['loss'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[93]:


# Getting the predicted stock price of 2017, Him
data_total = pd.concat((data_train['Adj Close'], data_holdout['Adj Close']), axis = 0)
inputs = data_total[len(data_total) - len(data_holdout) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
#array_holdout


# In[94]:


# Visualising the results
plt.plot(array_holdout, color = 'red', label = 'Real AAPL Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted AAPL Stock Price')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('AAPL Stock Price')
plt.legend()
plt.show()


# ANOTHER EXAMPLE: SOURCE MachineLearningMastery
# ANOTHER EXAMPLE: SOURCE MachineLearningMastery
# ANOTHER EXAMPLE: SOURCE MachineLearningMastery
# ANOTHER EXAMPLE: SOURCE MachineLearningMastery
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/

# univariate multi-step lstm
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM

# split a univariate dataset into train/test sets
def split_dataset(data):
	# split into standard weeks
	train, test = data[1:-328], data[-328:-6]
	# restructure into windows of weekly data
	train = array(split(train, len(train)/7))
	test = array(split(test, len(test)/7))
	return train, test

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=7):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end < len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)

# train the model
def build_model(train, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	verbose, epochs, batch_size = 0, 70, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# define model
	model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, len(input_x), 1))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat

# evaluate a single model
def evaluate_model(train, test, n_input):
	# fit model
	model = build_model(train, n_input)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores

# load the new file
dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
# split into train and test
train, test = split_dataset(dataset.values)
# evaluate model and get scores
n_input = 7
score, scores = evaluate_model(train, test, n_input)
# summarize scores
summarize_scores('lstm', score, scores)
# plot scores
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores, marker='o', label='lstm')
pyplot.show()
