# My LSTM Rewrite

# Imports are by section in prep for OOP refactoring

# ------------------- Get Data

# API
import os
import requests
# import json
# /API

# CSV
import pandas as pd
import numpy as np
def get_data(filename):
    '''Get CSV data
    '''
    return pd.read_csv(filename)
data = get_data('../data/AAPL.csv')
# /CSV

# ------------------- Split Data
def split_dataframe(data, percent_holdout=0.25):
    # implicit iloc dataframe reference
    index = round(len(data)*(1 - percent_holdout))
    data_train, data_test = data.iloc[:index, :], data.iloc[index:, :]
    return data_train, data_test
data_train, data_test = split_dataframe(data, 0.25)

# ------------------- Condition Data
# Flip data
data_train = data_train.iloc[::-1]
array_train = data_train.iloc[:, -1].values
# Detrend and reshape?
array_train = np.diff(array_train).reshape(-1, 1)
#array_train = array_train.reshape(-1, 1)
# Scale Data
# > If data is differenced, we assume constant variance in stock price
from sklearn.preprocessing import MinMaxScaler
def transform_scaled(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data), scaler
array_train_scaled, scaler = transform_scaled(array_train)

# Generate Sliding Window Train Data: Suggest a stationary TS
# [------------------------------]
# [---[------][----]-------------]
import numpy as np
def transform_slidingWindow(data, train_window, test_window, step = 1):
    x, y = [], []
    total_window = train_window + test_window
    slices_float = (len(data)-total_window)/step + 1
    slices, remainder = slices_float//1, slices_float%1
    for start in range(int(slices)):
        x.append(data[start*step : start*step + train_window])
        y.append(data[start*step + train_window : start*step + total_window])
    x = np.array(x)
    y = np.array(y)
    X = np.reshape(x, (x.shape[0], x.shape[1], 1))
    Y = np.reshape(y, (y.shape[0], y.shape[1]))
    if remainder > 0.01:
        print('Step Size Remainder! Excluding last {} days'.format(round(remainder*step)))
    return X, Y, X.shape, Y.shape

# Using 3 weeks, predict next week. We are swing trading
X_train, y_train, x_shape, y_shape = transform_slidingWindow(array_train_scaled, 21, 7)

def grab_debug(data, size = 3):
    return data[:size, :, :]

X_debug, y_debug = grab_debug(X_train), grab_debug(np.expand_dims(y_train, axis = 2)).squeeze()

# ------------------ Mind
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils.vis_utils import plot_model

def compose_mind(shape_of_x, shape_of_y):
    layer_in = {
        'units':200,
        'activation':'tanh',
        #'input_shape':(X_train.shape[1], 1)
        'input_shape':(shape_of_x[1], shape_of_x[2])
    }
    layer_hidden = {
        'units':25,
        'activation':'relu'
    }
    layer_output = {
        #'units':y_train.shape[1]
        'units':shape_of_y[1]
    }
    dropout = 0.3
    compile_params = {
        'optimizer':'adam',
        'loss':'mse'
    }
    mind = Sequential()
    mind.add(LSTM(**layer_in))
    mind.add(Dropout(dropout))
    mind.add(Dense(**layer_hidden))
    mind.add(Dense(**layer_output))
    mind.compile(**compile_params)
    print(mind.summary())
    return mind

mind = compose_mind(X_train.shape, y_train.shape)

# DEBUG BLOW
from datetime import datetime
from time import monotonic
model_name = '/LSTM_' + datetime.now().strftime('%m-%d_%H:%M')
fit_params = {
    'epochs':30,
    'batch_size':32,
    'shuffle':False
}
# Fitting

time_then = monotonic()
lstm1 = mind.fit(X_train, y_train, **fit_params, verbose = True)
time_now = monotonic()
# Summary things
mins_elapsed = (time_now - time_then)/60
# Save Model
print('Minutes Elapsed: ', mins_elapsed)

# ID Model and Save to Directory
import os
this_loc = '../resources/models'

with open(this_loc + model_name + '.json', 'w') as json_file:
    json_file.write(mind.to_json())
    mind.save_weights(this_loc + model_name + '.h5')
print('Model: ', model_name)
print('Time: ', mins_elapsed)
# --------------------- Evaluate
import matplotlib.pyplot as plt
def plot_training_loss(model_variable, model_name):
    fig, ax = plt.subplots()
    ax.plot(model_variable.history['loss'])
    ax.set_title('Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    fig.savefig(this_loc + model_name + 'loss.png', bbox_inches = 'tight')

plot_training_loss(lstm1, model_name)
plot_model(mind, to_file=this_loc + model_name + 'graph.png', show_shapes=True, show_layer_names=True)
# Test
# Flip, grab column of interest, as array, and reshape
array_test = data_test.iloc[::-1].iloc[:, -1].values.reshape(-1, 1)
array_test_scaled = scaler.transform(array_test)
X_test_transformed, y_test_transformed, _, _ = transform_slidingWindow(array_test_scaled, 21, 7, 7)
y_predicted = np.squeeze(scaler.inverse_transform(mind.predict(X_test_transformed).reshape(1, -1).astype('float')))
y_actual = scaler.inverse_transform(y_test_transformed.reshape(1, -1)).flatten()
residual = y_actual - y_predicted
fig, ax = plt.subplots()
ax.plot(list(y_predicted), '#32CD32', label = 'Predicted')
# Y test transformed is still a window
ax.plot(list(y_actual), '#00FF00', label = 'Actual')
ax.plot(list(-1*abs(residual)), '|', label = 'Abs. Residual')
ax.plot([0]*len(residual), '#B0C4DE', alpha = 0.8)
ax.legend(loc = 0, framealpha = 0.6)
ax.set_title('Predicted vs. Actual')
ax.set_xlabel('Days')
ax.set_ylabel('Change in Price')
fig.savefig(this_loc + model_name + 'PredVsActual.png', bbox_inches = 'tight')

# EXTRA CODE
