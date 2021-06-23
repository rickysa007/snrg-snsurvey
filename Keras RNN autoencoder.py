#!/usr/bin/env python
# coding: utf-8

# In[171]:


'''Line up the peak
Interpolation that taken account in the non-uniform timestep? --DONE
Feed the output back as input to see anything reasonable
Try multiple band?
See see the output from the encoding part
Philips relation from the output'''


# In[172]:


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


# In[173]:


# Import the .json file

os.chdir(r"C:\Users\ricky\JupyterNotebooks\Intern21\import_photometry_data\all_photometry")
filename = glob.glob('*.json')
print(len(filename))

# Create a list for all .json, the 1st SN saved as json_data[0], the 2nd SN saved as json_data[1], etc.
json_data = []
for i in filename:
    with open(i, encoding="utf-8") as f:
        json_data.append(json.load(f))


# In[174]:


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


# In[179]:


# Peak fitting

from sklearn.metrics import r2_score
Time_max = [] # Save the day of maximum magnitude
fitting_days = 30 # Num of days after the maximum used for peak fitting from the peak

for i in range(len(filename)): # Loop through all SN
    if len(Time[i]) > 0: # Avoid list with too few data
        maximum = np.argmin(Magnitude_Abs[i]) # Obtain the id of the maximum magnitude
        peak_time = Time[i][maximum] # Save the day of maximum magnitude
        tail_time = Time[i][maximum] # Save the day of the end of the initial lightcurve fall off
        
        # Calculate the day of the end of the initial lightcurve fall off
        j = 0
        if (peak_time + fitting_days) < Time[i][-1]: # Avoid light curve that is too short (fewer than 30 days after the peak)
            while tail_time < (peak_time + fitting_days):
                tail_time = Time[i][maximum + j]
                if tail_time > (peak_time + fitting_days):
                    break
                j += 1
        
        # Save the peaking part of the light curve
        m = Magnitude_Abs[i][:maximum + j]
        t = Time[i][:maximum + j]
        
        # Polynomial (degree = 3) fit of the peak part of the light curve 
        P = np.poly1d(np.polyfit(t, m, deg = 3))
        
        # Save the time of maximum magnitude
        Time_max.append(Time[i][np.argmin(P(t))])
        
print(Time_max)


# In[182]:


data = []
lightcurve_length_max = 0
lightcurve_num = 0
lightcurve_length = []
lightcurve_succ = []
lightcurve_days = 140


# To obtain individual lightcurve length (timesteps length) and the maximum lightcurve length
for i in range(len(filename)):
    
    if len(Time[i]) > 65: # Avoid lightcurve with too few data points
        if (Time[i][-1] - Time[i][0]) > lightcurve_days:
            for j in range(len(Time[i])):
                if (Time[i][j] - Time[i][0]) > lightcurve_days:
                    lightcurve_length.append(j)
                    lightcurve_length_max = max([j, lightcurve_length_max])
                    lightcurve_num += 1
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
        t = t_temp[0:lightcurve_length[j]+1] - t_temp[0]
        print('no. of data points is', len(t))
        print('the claimed type is', Type[i])
        m = Magnitude_Abs[i][0:lightcurve_length[j]+1]
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

data = np.array(data)
lightcurve = data.reshape(lightcurve_num, steps, 1) #no. of sample (batch size), timesteps in RNN, no. of features

print(lightcurve.shape)
print(lightcurve)


# In[165]:


# define model
model = Sequential()
model.add(GRU(150, activation='tanh', input_shape=(steps,1), return_sequences=True))
model.add(GRU(50, activation='tanh', return_sequences=True))
model.add(GRU(10, activation='tanh', return_sequences=False))
model.add(RepeatVector(steps))
model.add(GRU(50, activation='tanh', return_sequences=True))
model.add(GRU(150, activation='tanh', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
model.summary()
plot_model(model, show_shapes=True)


# In[166]:


# fit model
callbacks = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, restore_best_weights=True)
history = model.fit(lightcurve, lightcurve, validation_split = 0.25, epochs=1500, verbose=2)


# In[168]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.xlim(0, 1500)
plt.ylim(0, 1)


# In[169]:


# demonstrate recreation
yhat = model.predict(lightcurve, verbose=2)
yhat1 = model.predict(yhat, verbose=2)
#print(yhat[0,:,0])


# In[170]:


j = 0

for i in range(len(filename)):
    if lightcurve_succ[i] == 1:
        fig = plt.figure(figsize=(16,10))
        ax = fig.add_subplot(1, 1, 1)

        plt.gca().invert_yaxis()

        # And a corresponding grid
        ax.grid(which='major', alpha=0.8)
        ax.grid(which='minor', alpha=0.3)

        plt.xlabel('Timestep')
        plt.ylabel('Absolute Magnitude')

        x = np.linspace(1, steps, steps)

        plt.scatter(x, lightcurve[j,:,0], s=2)
        plt.scatter(x, yhat[j,:,0], s=2)
        #plt.scatter(x, yhat1[j,:,0], s=2)
        
        print('the claimed type is', Type[i])
        
        plt.show()
        
        j += 1

