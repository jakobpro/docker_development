from enum import auto
import os
from datetime import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.api import OLS
from statsmodels.tsa.stattools import adfuller

from utilities.statistics import distribution
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.stattools import durbin_watson

plt.switch_backend('agg')

def exploration(df, timeName, labelName,frequency):    
    print(df.head())
    print(timeName, labelName)
    print('Exploration started')
    
    # Get general description
    description = st.describe(df[labelName])
    print(description)
    
    # Get distribution plot
    ax = sns.distplot(df[labelName],
                  kde=True,
                  color='orange')
    
    fig = ax.get_figure()
    plt.savefig('distplot.jpg') 
    
    # get best distribution
    best_dist, best_p, result_dict = distribution(df[labelName].dropna()) 
    print(best_dist, best_p, result_dict )
    
    df[timeName] = pd.to_datetime(df[timeName])
    df = df.set_index(timeName).asfreq(frequency) ## Bei einer aggregation stimmt das hier noch nicht ganz!
    df = df.fillna(axis = 0,method='ffill')
    
    # Seasonal decomposition
    plt.rcParams.update({'figure.figsize': (10,10)})
    # Additive Decomposition
    result_add = seasonal_decompose(df, model='additive')
    result_add.plot().suptitle('Additive Decompose', fontsize=16)
    plt.savefig("seasonal_decomposition_additive.jpg")
    
    # Multiplicative Decomposition 
    if min(df[labelName]) > 0:
        result_mul = seasonal_decompose(df, model='multiplicative' )
        result_mul.plot().suptitle('Multiplicative Decompose', fontsize=16)
        plt.savefig("seasonal_decomposition_multiplicative.jpg")
    
    # OLS Regression
    x, y = np.arange(len(result_add.trend.dropna())), result_add.trend.dropna()
    x = sm.add_constant(x)
    model = OLS(y, x)
    res = model.fit()
    print(res.summary())
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    ax[0].plot(result_add.trend.dropna().values, label='trend')
    ax[0].plot([res.params.x1*i + res.params.const for i in np.arange(len(result_add.trend.dropna()))])
    ax[1].plot(res.resid.values)
    ax[1].plot(np.abs(res.resid.values))
    ax[1].hlines(0, 0, len(res.resid), color='r')
    ax[0].set_title("Trend and Regression")
    ax[1].set_title("Residuals")
    plt.savefig("regression_trend.jpg")
    
    # adfuller Test
    r = adfuller(df, autolag='AIC')
    print(f'Test Statistics: {r[0]}')
    print(f'P-value: {r[1]}')
    if abs(r[1]) < 0.05:
        print('Null hypothesis is rejected! Time series is stationary')
    else:
        print('Null hypothesis is accepted! Stationarity cannot be proven')
        
    # autocorrelation test
    autocorrelation = durbin_watson(res.resid)

    if autocorrelation < 1.5 or autocorrelation > 2.5:
        print('autocorrelation detected!')
        if autocorrelation < 1.5 and autocorrelation > 0.5:
            print('Positive serial correlation')
            autocor = {'Positive serial correlation': autocorrelation}
        if autocorrelation < 0.5:
            print('Strong positive serial correlation')
            autocor = {'Strong positive serial correlation': autocorrelation}
        if autocorrelation > 2.5 and autocorrelation < 3.5:
            print('Negative serial correlation')
            autocor = {'Negative serial correlation': autocorrelation}
        if autocorrelation > 3.5:
            print('Strong negative serial correlation')
            autocor = {'Strong negative serial correlation': autocorrelation}
    else: 
        print('No serial correlation.')
        
    # upload file with settings 
    info_file = os.path.join('Statistics.txt')

    with open(info_file, 'w') as file:
        file.write('Explorative Analysis:')
        file.write('\n\n==============================================================================')
        file.write('\n\nStatistical Description:')
        file.write('\nNumber of Observations: {}'.format(description[0]))
        file.write('\nMin/Max Values: {}'.format(description[1]))
        file.write('\nMean: {}'.format(description[2]))
        file.write('\nVariance: {}'.format(description[3]))
        file.write('\nSkewness: {}'.format(description[4]))
        file.write('\Kurtosis: {}'.format(description[5]))
        file.write('\n\n==============================================================================')
        file.write('\n\nDistribution Metrics:')
        file.write('\nPredicted Distribution: {}'.format(best_dist))
        file.write('\nP-Value: {}'.format(best_p))
        file.write('\nComplete Distribution Results: {}'.format(result_dict))
        file.write('\n\n==============================================================================')
        file.write('\n\nTrend Analysis:')
        file.write('\nOLS Regression Summary: {}'.format(res.summary()))
        file.write('\n\n==============================================================================')
        file.write('\n\nStationarity with Adfuller-Test:')
        file.write('\nTest Statistics: {}'.format(r[0]))
        file.write('\nP-Value: {}'.format(r[1]))
        file.write('\nMost probable Seasonal Periodicity (Lag): {}'.format(r[2]))
        if abs(r[1]) < 0.05:
            file.write('\nNull hypothesis is rejected! Time series is stationary')
        else:
            file.write('\nNull hypothesis is accepted! Stationarity cannot be proven')
        file.write('\n\nAutocorrelation Test (Durbin-Watson):')  
        file.write('\nSerial Correlation: {}'.format(autocor))
        
        
