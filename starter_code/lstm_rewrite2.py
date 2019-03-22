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
# Scale Data
data = data.iloc[::-1]
data_array = data.iloc[:, -1].values
# Detrend and reshape?
data_array = np.diff(data_array).reshape(-1, 1)
#array_train = array_train.reshape(-1, 1)
# > If data is differenced, we assume constant variance in stock price
from sklearn.preprocessing import MinMaxScaler
def transform_scaled(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data), scaler
data_array_scaled, scaler = transform_scaled(data_array)
# ------------------- Split Data
def split_data(data, percent_holdout=0.25):
    # implicit iloc dataframe reference
    index = round(len(data)*(1 - percent_holdout))
    data_train, data_test = data[:index, :], data[index:, :]
    return data_train, data_test
array_train, array_test = split_data(data_array_scaled, 0.25)

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
X_train, y_train, x_shape, y_shape = transform_slidingWindow(array_train, 21, 7)

def grab_debug(data, size = 3):
    return data[:size, :, :]

X_debug, y_debug = grab_debug(X_train), grab_debug(np.expand_dims(y_train, axis = 2)).squeeze()

# ------------------ Mind
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils.vis_utils import plot_model

def compose_mind(shape_of_x, shape_of_y):
    layer_in = {
        'units':100,
        'activation':'tanh',
        'input_shape':(shape_of_x[1], shape_of_x[2])
    }
    # layer_hidden_lstm = {
    #     'units':50,
    #     'activation':'tanh'
    # }
    layer_hidden = {
        'units':100,
        'activation':'relu'
    }
    layer_hidden2 = {
        'units':20,
        'activation':'relu'
    }
    layer_output = {
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
    # mind.add(LSTM(**layer_hidden_lstm))
    # mind.add(Dropout(dropout))
    mind.add(Dense(**layer_hidden))
    mind.add(Dense(**layer_hidden2))
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
X_test_transformed, y_test_transformed, _, _ = transform_slidingWindow(array_test, 21, 7, 7)
y_predicted = np.squeeze(scaler.inverse_transform(mind.predict(X_test_transformed).reshape(1, -1).astype('float')))
y_actual = scaler.inverse_transform(y_test_transformed.reshape(1, -1)).flatten()
residual = y_actual - y_predicted

fig, ((ax, bx),(cx, dx)) = plt.subplots(nrows = 2, ncols=2, figsize = (10, 10))
def plot_zoom(ax, ind1, ind2):
    ax.plot(list(y_predicted[ind1:ind2]), '#32CD32', label = 'Predicted')
    # Y test transformed is still a window
    ax.plot(list(y_actual[ind1:ind2]), '#00FF00', label = 'Actual')
    ax.plot(list(-1*abs(residual[ind1:ind2])), '|', label = 'Abs. Residual')
    ax.plot([0]*len(residual[ind1:ind2]), '#B0C4DE', alpha = 0.8)
    ax.legend(loc = 0, framealpha = 0.6)
    ax.set_title('Pred vs. Actual: {}-{}'.format(ind1, ind2))
    ax.set_xlabel('Days')
    ax.set_ylabel('Change in Price')

plot_zoom(ax, 200, 242)
plot_zoom(bx, 100, 142)
plot_zoom(cx, 0, 200)
plot_zoom(dx, 800, 900)

fig.savefig(this_loc + model_name + 'PredVsActual_Zoom.png', bbox_inches = 'tight')

# EXTRA CODE

fig, ax = plt.subplots()
ax.plot(np.cumsum(y_predicted), 'y', label = 'Predicted')
# Y test transformed is still a window
ax.plot(np.cumsum(y_actual), 'k', label = 'Actual')
ax.legend(loc = 0, framealpha = 0.6)
ax.set_title('Pred vs. Actual: Cumulative Sum')
ax.set_xlabel('Days')
ax.set_ylabel('CS Price')
fig.savefig(this_loc + model_name + 'CumSum.png', bbox_inches = 'tight')
