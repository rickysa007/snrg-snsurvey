#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''To Do List
change to .json to get the Luminosity distance --DONE,
use polynomial fit --DONE,
show some graphs of fitting --DONE,
show differnce in fitting graph,
type IIP supernova,
Philips relation'''


# In[2]:


import os
import glob
import numpy as np
import pandas as pd
import json
import torch
import matplotlib.pyplot as plt


# In[3]:


# Import the .json file

os.chdir(r"C:\Users\ricky\JupyterNotebooks\Intern21\import_photometry_data\typeIIP_photometry")
filename = glob.glob('*.json')
#print(filename)

# Create a list for all .json, the 1st SN saved as json_data[0], the 2nd SN saved as json_data[1], etc.
json_data = []
for i in filename:
    print(i)
    with open(i, encoding="utf-8") as f:
        json_data.append(json.load(f))


# In[4]:


# To obtain absolute magnitude and time in a particular band

Band = [] # Contain EM band chosen for analysis
Magnitude_Abs = [] # Contain absolute magnitude
Time = [] # Contain time (day)

print(len(filename))

for i in range(len(filename)): # Loop through all SN
    Band.append([]) # Create 2D list
    Magnitude_Abs.append([])
    Time.append([])
    
    SN_name = filename[i].replace('.json', '')
    SN_name = SN_name.replace('_', ':')
    print(SN_name)
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

'''print(Band[13])
print(Magnitude_Abs[13])
print(len(Time[13]))'''


# In[9]:


# Peak fitting

from sklearn.metrics import r2_score
Succ_graph = [] # Save the number of successfully fitted graph
Time_max = [] # Save the day of maximum magnitude
fitting_days = 30 # Num of days after the maximum used for peak fitting from the peak

for i in range(len(filename)): # Loop through all SN
    if len(Time[i]) > 60: # Avoid list with too few data
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
        
        print('id:', i, ', SN:', filename[i], ', R^2 score:', r2_score(m, P(t)), ', Time of maximum brightness:', Time_max[i])        
        
        # Print the graph of fitting
        X = np.linspace(Time[i][0], Time[i][maximum + j], 200)
        Y = P(X)
        plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel('Time (day)')
        plt.ylabel('Absolute Magnitude')
        plt.xlim(Time[i][0] - 10, Time[i][maximum + j] + 100)
        plt.scatter(X, Y, s=2)
        plt.scatter(Time[i], Magnitude_Abs[i], s=2)
        plt.show()
        
        # Print the graph of difference in fitting
        plt.scatter(t, abs(m - P(t)), s=2)
        plt.grid()
        plt.show()
    
        if r2_score(m, P(t)) > 0.8:
            Succ_graph.append(1)
        else:
            Succ_graph.append(0)
        
    else:
        Time_max.append(0)
        Succ_graph.append(0)
        
print(Succ_graph)


# In[10]:


Time_shifted = [] # Save the shifted time list for better result demonstration (All light curve peaking at day 0)
k = 0

for i in range(len(Succ_graph)):
    Time_shifted.append([]) # Create 2D list
    if Succ_graph[i] == 1:
        diff = Time_max[i] - Time_max[0] 
        for j in range(len(Time[i])): # Calculate the shifted time list
            Time_shifted[i].append(Time[i][j] - diff - Time_max[0])
            
        k+=1

print(k)


# In[14]:


fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(1, 1, 1)

plt.gca().invert_yaxis()

# Major ticks every 20, minor ticks every 5
major_ticks_x = np.arange(-50, 600, 50)
minor_ticks_x = np.arange(-50, 600, 10)

major_ticks_y = np.arange(-22, -5, 2)
minor_ticks_y = np.arange(-22, -5, 1)


ax.set_xticks(major_ticks_x)
ax.set_xticks(minor_ticks_x, minor=True)

ax.set_yticks(major_ticks_y)
ax.set_yticks(minor_ticks_y, minor=True)

# And a corresponding grid
ax.grid(which='major', alpha=0.8)
ax.grid(which='minor', alpha=0.3)

plt.xlabel('Time (day)')
plt.ylabel('Absolute Magnitude')

plt.xlim(-20, 250)
plt.ylim(-5, -19)

for i in range(len(Succ_graph)):
    if Succ_graph[i] == 1:
        plt.scatter(Time_shifted[i], Magnitude_Abs[i], s=2)

