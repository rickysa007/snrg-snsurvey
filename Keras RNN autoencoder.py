#!/usr/bin/env python
# coding: utf-8

# In[272]:


'''Line up the peak
Interpolation that taken account in the non-uniform timestep? --DONE
Feed the output back as input to see anything reasonable
Try multiple band?
See see the output from the encoding part
Philips relation from the output'''


# In[273]:


import os
import glob
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d, splrep

from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping
from keras.utils import plot_model


# In[290]:


# Import the .json file

os.chdir(r"C:\Users\ricky\JupyterNotebooks\Intern21\import_photometry_data\all_photometry")
filename = glob.glob('*.json')
np.random.shuffle(filename)
print(filename)

'''x = numpy.random.rand(100, 5)
numpy.random.shuffle(x)
training, test = x[:80,:], x[80:,:]'''

# Create a list for all .json, the 1st SN saved as json_data[0], the 2nd SN saved as json_data[1], etc.
json_data = []
for i in filename:
    with open(i, encoding="utf-8") as f:
        json_data.append(json.load(f))


# In[291]:


# To obtain absolute magnitude and time in a particular band

Band = [] # Contain EM band chosen for analysis
Magnitude_Abs = [] # Contain absolute magnitude
Time = [] # Contain time (day)
Type = [] # Claimed type

for i in range(len(filename)): # Loop through all SN
    Band.append([]) # Create 2D list
    Magnitude_Abs.append([])
    Time.append([])
    
    SN_name = filename[i].replace('.json', '')
    SN_name = SN_name.replace('_', ':')
    
    Type.append(json_data[i][SN_name]['claimedtype'][0]['value'])
    
    N = len(json_data[i][SN_name]['photometry']) # The no. of data point of photometry in each SN
    
    for j in range(N): # Loop through all photemetry datapoint in one SN
        # Avoid any data point without band data
        try:
            Band[i].append(json_data[i][SN_name]['photometry'][j]['band'])
        except:
            Band[i].append(0)
        
        # Fill the Magnitude_Abs and Time list if the data point is in B band
        if Band[i][j] == 'B':
            Magnitude_App = float(json_data[i][SN_name]['photometry'][j]['magnitude']) # Obtain the apparent magnitude from photometry
            LumDist = float(json_data[i][SN_name]['lumdist'][0]['value']) # Obtain the luminosity distance
            z = float(json_data[i][SN_name]['redshift'][0]['value']) #Obtain the redshift, z
            Magnitude_Abs[i].append(Magnitude_App - 5*np.log10(LumDist*1e5) + 2.5*np.log10(1+z)) # Calculate the absolute magnitude and fill the Magnitude_Abs list
            Time[i].append(float(json_data[i][SN_name]['photometry'][j]['time'])) # Fill the Time list


# In[299]:


# Interpolating the data

data = []
lightcurve_length_max = 0
lightcurve_length = []
lightcurve_succ = []
lightcurve_days = 130


# To obtain individual lightcurve length (timesteps length) and the maximum lightcurve length
for i in range(len(filename)):
    
    if len(Time[i]) > 65: # Avoid lightcurve with too few data points
        if (Time[i][-1] - Time[i][0]) > lightcurve_days:
            for j in range(len(Time[i])):
                if (Time[i][j] - Time[i][0]) > lightcurve_days:
                    lightcurve_length.append(j)
                    lightcurve_length_max = max([j, lightcurve_length_max])
                    lightcurve_succ.append(1)
                    break
        else:
            lightcurve_succ.append(0)
    else:
        lightcurve_succ.append(0)

print(lightcurve_length_max)
print(len(filename))
print(len(lightcurve_succ))
print(lightcurve_succ)
print(len(lightcurve_length))
print(lightcurve_length)

j = 0

steps = 150

for i in range(len(filename)):
    
    if lightcurve_succ[i] == 1:
        t_temp = np.array(Time[i]) 
        t = t_temp[:lightcurve_length[j]+1] - t_temp[0]
        print('no. of data points is', len(t))
        print('the claimed type is', Type[i])
        m = Magnitude_Abs[i][:lightcurve_length[j]+1]
        f = interp1d(t, m)
        tnew = np.linspace(0, lightcurve_days, steps)
        mnew = f(tnew)
        
        fig = plt.figure(figsize=(16,10))
        plt.gca().invert_yaxis()
        plt.grid()
        plt.scatter(t, m, s=2)
        plt.scatter(tnew, mnew, s=2)
        plt.show()
        
        data.append(mnew)
        j += 1


# In[302]:


# Spliting training set and testing set

split_portion = 0.8

data = np.array(data)
split = int(split_portion*len(data))
data_train = data[:split]
data_test = data[split:]
lightcurve_train = data_train.reshape(len(data_train), steps, 1) #no. of sample (batch size), timesteps in RNN, no. of features
lightcurve_test = data_test.reshape(len(data_test), steps, 1)

print(lightcurve_train.shape)
print(lightcurve_test.shape)


# In[303]:


# define model
model = Sequential()
model.add(GRU(150, activation='tanh', input_shape=(steps,1), return_sequences=True))
model.add(GRU(40, activation='tanh', return_sequences=True))
model.add(GRU(10, activation='tanh', return_sequences=False))
model.add(RepeatVector(steps))
model.add(GRU(40, activation='tanh', return_sequences=True))
model.add(GRU(150, activation='tanh', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
model.summary()
plot_model(model, show_shapes=True)


# In[304]:


# Fit model
callbacks = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=20, restore_best_weights=True)
history = model.fit(lightcurve_train, lightcurve_train, validation_split = 0.2, epochs=1500, verbose=2)


# In[305]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.xlim(0, 1500)
plt.ylim(0, 1)


# In[306]:


# Delete the training set to save some ram
'''del(lightcurve_train)
del(data_train)'''

# Demonstrate recreation
yhat = model.predict(lightcurve_test, verbose=2)
yhat1 = model.predict(yhat, verbose=2)
#print(yhat[4,:,0])


# In[307]:


j = 0

for i in range(len(data_test)):
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(1, 1, 1)

    plt.gca().invert_yaxis()

    # And a corresponding grid
    ax.grid(which='major', alpha=0.8)
    ax.grid(which='minor', alpha=0.3)

    plt.xlabel('Timestep')
    plt.ylabel('Absolute Magnitude')

    x = np.linspace(1, steps, steps)

    plt.scatter(x, lightcurve_test[j,:,0], s=2)
    plt.scatter(x, yhat[j,:,0], s=2)
    #plt.scatter(x, yhat1[j,:,0], s=2)
        
    print('the claimed type is', Type[i])
        
    plt.show()
        
    j += 1

