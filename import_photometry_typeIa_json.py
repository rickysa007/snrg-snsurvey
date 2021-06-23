#!/usr/bin/env python
# coding: utf-8

# In[12]:


'''To Do List
change to .json to get the Luminosity distance --DONE,
use polynomial fit --DONE,
show some graphs of fitting --DONE,
show differnce in fitting graph,
type IIP supernova,
Philips relation'''


# In[13]:


import os
import glob
import numpy as np
import pandas as pd
import json
import math as mth
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit, minimize
from lmfit import Model


# In[14]:


# Import the .json file

os.chdir(r"C:\Users\ricky\JupyterNotebooks\Intern21\import_photometry_data\typeIa_photometry")
filename = glob.glob('*.json')
print(len(filename))

# Create a list for all .json, the 1st SN saved as json_data[0], the 2nd SN saved as json_data[1], etc.
json_data = []
for i in filename:
    with open(i, encoding="utf-8") as f:
        json_data.append(json.load(f))


# In[15]:


'''import mosfit

# Create an instance of the `Fetcher` class.
my_fetcher = mosfit.fetcher.Fetcher()

# Fetch some data from the Open Supernova Catalog.
fetched = my_fetcher.fetch('SN1994D')[0]

# Instantiatiate the `Model` class (selecting 'slsn' as the model).
my_model = mosfit.model.Model(model='default')

# Load the fetched data into the model.
my_model.load_data(my_fetcher.load_data(fetched), event_name=fetched['name'])

# Generate a random input vector of free parameters.
x = np.random.rand(my_model.get_num_free_parameters())

# Produce model output.
outputs = my_model.run(x)
print('Keys in output: `{}`'.format(', '.join(list(outputs.keys()))))'''


# In[16]:


# To obtain absolute magnitude and time in a particular band

Band = [] # Contain EM band chosen for analysis
Magnitude_Abs = [] # Contain absolute magnitude
Time = [] # Contain time (day)

for i in range(len(filename)): # Loop through all SN
    Band.append([]) # Create 2D list
    Magnitude_Abs.append([])
    Time.append([])
    
    SN_name = filename[i].replace('.json', '')
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


# In[17]:


# Peak fitting

from sklearn.metrics import r2_score
Succ_graph = [] # Save the number of successfully fitted graph
Time_max = [] # Save the day of maximum magnitude
fitting_days = 15 # Num of days after the maximum used for peak fitting from the peak

for i in range(len(filename)): # Loop through all SN
    if len(Time[i]) != 0: # Avoid empty list
        maximum = np.argmin(Magnitude_Abs[i]) # Obtain the id of the maximum magnitude
        peak_time = Time[i][maximum] # Save the day of maximum magnitude
        tail_time = Time[i][maximum] # Save the day of the end of the initial lightcurve fall off
        
        # Calculate the day of the end of the initial lightcurve fall off
        j = 0
        if (peak_time + fitting_days) < Time[i][-1]: # Avoid light curve that is too short (fewer than 15 days after the peak)
            while tail_time < (peak_time + fitting_days):
                tail_time = Time[i][maximum + j]
                if tail_time > (peak_time + fitting_days):
                    break
                j += 1
        
        # Save the peaking part of the light curve
        t = Time[i][:maximum + j]
        m = Magnitude_Abs[i][:maximum + j]
        
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
        #plt.xlim(Time[i][0] - 10, Time[i][maximum + j] + 50)
        plt.scatter(X, Y, s=2)
        plt.scatter(Time[i], Magnitude_Abs[i], s=2)
        plt.show()
    
        if r2_score(m, P(t)) > 0.8:
            Succ_graph.append(1)
        else:
            Succ_graph.append(0)
        
    else:
        Time_max.append(0)
        Succ_graph.append(0)


print(Succ_graph)


# In[18]:


'''lamb = 0.5
sigma = 0.14
offset = 0.5
def expconv(xrange):    
    piece1 = 1/(2*lamb)
    piece2 = np.exp(sigma*sigma/(2*lamb*lamb) -(xrange-offset)/lamb)
    piece3 = mth.erf(sigma/((np.sqrt(2)*lamb))-(xrange-offset)/(np.sqrt(2)*sigma))
    return np.log10(piece1*piece2*(1.0-piece3))

vec_expconv = np.vectorize(expconv)
xrange = np.arange(0.0,4, 0.01)
plt.plot(xrange, vec_expconv(xrange))
plt.show()         
'''


# In[27]:


def expconv(xrange, lamb, sigma, offset, A):    
    piece1 = A/(2*lamb)
    piece2 = np.exp(sigma*sigma/(2*lamb*lamb) - (xrange-offset)/lamb)
    piece3 = sp.special.erfc(sigma/((np.sqrt(2)*lamb)) - (xrange-offset)/(np.sqrt(2)*sigma))
    print(piece1*piece2*piece3)
    return np.log10(piece1*piece2*piece3)

def cost(x, x_axis, datapoint_value):
    return np.sum((expconv(x_axis, x[0], x[1], x[2], x[3]) - datapoint_value)**2)

# Peak fitting

from sklearn.metrics import r2_score
Succ_graph = [] # Save the number of successfully fitted graph
Time_max = [] # Save the day of maximum magnitude
fitting_days = 15 # Num of days after the maximum used for peak fitting from the peak

for i in range(len(filename)): # Loop through all SN
    if len(Time[i]) != 0: # Avoid empty list
        maximum = np.argmin(Magnitude_Abs[i]) # Obtain the id of the maximum magnitude
        peak_time = Time[i][maximum] # Save the day of maximum magnitude
        tail_time = Time[i][maximum] # Save the day of the end of the initial lightcurve fall off
        
        # Calculate the day of the end of the initial lightcurve fall off
        j = 0
        if (peak_time + fitting_days) < Time[i][-1]: # Avoid light curve that is too short (fewer than 15 days after the peak)
            while tail_time < (peak_time + fitting_days):
                tail_time = Time[i][maximum + j]
                if tail_time > (peak_time + fitting_days):
                    break
                j += 1
        
        # Save the peaking part of the light curve
        t = Time[i][:maximum + j]
        m = Magnitude_Abs[i][:maximum + j]
        
        lamb_guess = 7
        sigma_guess = 5
        offset_guess = Time[i][maximum]
        A_guess = 1
        guesses = [lamb_guess, sigma_guess, offset_guess, A_guess]
        
        
        
        
        debug2 = [np.exp(sigma_guess*sigma_guess/(2*lamb_guess*lamb_guess) - (t[j]-offset_guess)/lamb_guess) for j in range(len(t))]
        print('piece 2 is', debug2)
        debug3 = [sp.special.erfc(sigma_guess/((np.sqrt(2)*lamb_guess)) - (t[j]-offset_guess)/(np.sqrt(2)*sigma_guess)) for j in range(len(t))]
        print('piece 3 is', debug3)
        for j in range(len(debug2)):
            print(np.log10((A_guess/(2*lamb_guess))*debug2[j]*debug3[j]))
        
        
        
        
        res = minimize(cost, guesses, args = (t, m), method = 'COBYLA', bounds = ((0, 20), (0, 20), (offset_guess-10, offset_guess+10), (-100, 100)))
        print(res)
        print(expconv(t, res.x[0], res.x[1], res.x[2], res.x[3]))
        
        # Save the time of maximum magnitude
        Time_max.append(res.x[2])
        
        print('id:', i, ', SN:', filename[i], ', R^2 score:', r2_score(m, expconv(t, res.x[0], res.x[1], res.x[2], res.x[3])), ', Time of maximum brightness:', Time_max[i])
        
        # Print the graph of fitting
        X = np.linspace(Time[i][0], Time[i][maximum + j], len(t))
        Y = expconv(t, res.x[0], res.x[1], res.x[2], res.x[3])
        plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel('Time (day)')
        plt.ylabel('Absolute Magnitude')
        plt.xlim(Time[i][0] - 10, Time[i][maximum + j] + 50)
        plt.scatter(X, Y, s=2)
        plt.scatter(Time[i], Magnitude_Abs[i], s=2)
        plt.show()
    
        if r2_score(m, expconv(t, res.x[0], res.x[1], res.x[2], res.x[3])) > 0.8:
            Succ_graph.append(1)
        else:
            Succ_graph.append(0)
        
    else:
        Time_max.append(0)
        Succ_graph.append(0)


print(Succ_graph)


# In[20]:


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


# In[21]:


fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(1, 1, 1)

plt.gca().invert_yaxis()

# Major ticks every 20, minor ticks every 5
major_ticks_x = np.arange(-50, 600, 50)
minor_ticks_x = np.arange(-50, 600, 10)

major_ticks_y = np.arange(-22, -6, 2)
minor_ticks_y = np.arange(-22, -6, 1)


ax.set_xticks(major_ticks_x)
ax.set_xticks(minor_ticks_x, minor=True)

ax.set_yticks(major_ticks_y)
ax.set_yticks(minor_ticks_y, minor=True)

# And a corresponding grid
ax.grid(which='major', alpha=0.8)
ax.grid(which='minor', alpha=0.3)

plt.xlabel('Time (day)')
plt.ylabel('Absolute Magnitude')

plt.xlim(-20, 200)
#plt.ylim(-12, -21)

for i in range(len(Succ_graph)):
    if Succ_graph[i] == 1:
        plt.scatter(Time_shifted[i], Magnitude_Abs[i], s=2)


# In[25]:


# Philips relation

M = []
dm_P = []

for i in range(len(filename)): # Loop through all SN
    if Succ_graph[i] == 1: # Avoid poor fitting curve
        maximum = np.argmin(Magnitude_Abs[i]) # Obtain the id of the maximum magnitude
        peak_time = Time[i][maximum] # Save the day of maximum magnitude
        tail_time = Time[i][maximum] # Save the day of the end of the initial lightcurve fall off
        
        # Calculate the day of the end of the initial lightcurve fall off
        j = 0
        if (peak_time + fitting_days) < Time[i][-1]: # Avoid light curve that is too short (fewer than 15 days after the peak)
            while tail_time < (peak_time + fitting_days):
                tail_time = Time[i][maximum + j]
                if tail_time > (peak_time + fitting_days):
                    break
                j += 1
        
            # Save the peaking part of the light curve
            t = Time[i][:maximum + j]
            m = Magnitude_Abs[i][:maximum + j]
        
            # Polynomial (degree = 3) fit of the peak part of the light curve 
            P = np.poly1d(np.polyfit(t, m, deg = 3))
        
            a = t[-1] - t[maximum]
            b = P(t[-1]) - P(t[maximum])
        
            M.append(P(t[maximum]))
            dm_P.append(15 * b / a)


P1 = np.poly1d(np.polyfit(dm_P, M, deg = 1))
print(P1)
print(r2_score(M, P1(dm_P)))
#print(len(m_P), len(M))


# Plot out the linear relationship
plt.gca().invert_yaxis()
plt.xlabel('$\Delta m_{15}$')
plt.ylabel('Maximum absolute magnitude, $M_{max}$')
x = np.linspace(0, 2.5, 100)
plt.xlim(0.5, 2)
plt.scatter(x, P1(x), s=3)
plt.scatter(dm_P, M, s=3)

