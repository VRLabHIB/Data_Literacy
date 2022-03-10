# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:27:22 2020

Library with some major functions for processing the signal of the pupil 
diameter of both eyes, recorded with a standard Eye-Tracker.

"""
__version__='0.2'
__author__='Philipp Stark'

import os

def help():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    f = open(dir_path+'\help.txt', 'r')
    file_contents = f.read()
    print (file_contents)
    f.close()

import matplotlib.pyplot as plt 
import numpy as np
import scipy.signal as ss
import scipy.ndimage as sn

class Check():
    import numpy as np

    def tracking_ratio(df=None, variable=None, missings=None, thresholds=None):
        """
        Computes the tracking ration for a time dependent signal as ratio of valid and invalid points

        Parameters
        ----------
        df::py:class: 'pandas.DataFrame'
            time series i.e. df.loc[time, variable]

        missings: `numpy.ndarray`
            array of missing values which should be counted as invalid

        thresholds: `numpy.ndarray`
            lower and upper bound for signal i.e. [min , max]

        Returns
        -------
        tracking_ratio: 'float'
            percentage of valid data points in the time series
            
        """

        return None

class Process():
       
    def dilation_speed(df, VAR):
        """
        For each value in the dataframe, takes the maximum of the dilation speed 
        between VAR_t and VAR_t+1 or VAR_t and VAR_t-1
        
        Parameters
        ----------
        df : Pandas DataFrame
            Contains variables: 'time' and VAR.
        VAR : String
            Variable to calculate dilation from.

        Returns
        -------
        d_speed = max dilation speed

        """
        
        #calculate dilation speed
        arr = df[VAR].values
        time = df['time'].values
    
        d00 = arr[:-1]
        d01 = np.zeros(len(arr))
        d01[1:] = d00
    
        t00 = time[:-1]
        t01 = np.zeros(len(time))
        t01[1:] = t00
    
        d10 = arr[1:]
        d11 = np.zeros(len(arr))
        d11[:-1] = d10
    
        t10 = time[1:]
        t11 = t01 = np.zeros(len(time))
        t11[:-1] = t10
    
        speed01 = [np.abs((arr-d01) / (time-t01))]
        speed01[0][0] = 0
        speed02 = [np.abs((d11-arr) / (t11-time))]
        speed02[0][-1] = 0
        d_speed = np.max(np.concatenate((speed01,speed02),axis=0), axis=0)
    
        return d_speed
    
    def MAD_threshold(df, VAR,n):
        """
        Median threshold with a free parameter n. Note: Smaller n removes more 
        values. 

        Parameters
        ----------
        df : Pandas DataFrame
            Contains variables: VAR.
        VAR : String
            Name of the variable in df.
        n : Int
            Threshold Hyperparameter.

        Returns
        -------
        trsh : Float
            MAD threshold to remove dialtion speed outlier above the threshold.

        """
        med = np.nanmedian(df[VAR])
        MAD = np.nanmedian(np.abs(df[VAR]-med))
        trsh = n*MAD + np.nanmedian(df[VAR])
        return trsh

def smooth_and_filter(x,window_len=11,window='hanning',poly_order=3):
    """smooth and filter the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    NOTE: poly_order = polynomial order is only needed for the savgol smoothing
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal or filtered (only with median filter)
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    scipy.ndimage, median, scipy.signal.savgol, numpy.hanning, numpy.hamming, 
    numpy.bartlett, numpy.blackman, numpy.convolve, scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['median', 'savgol','flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'median', 'savgol', 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    if window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
        
        if window == 'flat': #moving average
            w = np.ones(window_len,'d')
        else:
            w = eval('np.' + window + '(window_len)')
    
        y = np.convolve(w/w.sum(),s,mode='valid')
        y = y[(window_len/2-1):-(window_len/2)]
        
    else:
        if window =='median':
            y = sn.median_filter(x, size = window_len)
        else:
            y = ss.savgol_filter(x, window_len, poly_order)
            
    return y        
    

class Display():
    
    def plot_pupil(df1, df2):
        """
        Parameters
        ----------
        df1 : Pandas DataFrame
            Contains variables: 'time' and 'left'.
        df2 : Pandas DataFrame
            Contains variables: 'time' and 'right'.

        Returns
        -------
        2 plots of the pupil diameter for left and right eye
        """
        fig = plt.figure(figsize=(15,10))
        y1 = df1['left'].values
        y2 = df2['right'].values
        x1 = df1['time'].values
        x2= df2['time'].values
        plt.subplot(211)
        plt.title('Left pupil')
        plt.scatter(x1,y1, s=0.1,c='grey')
        plt.xlim(0,x1[-1])
        plt.xlabel('time')
        plt.ylabel('pupil diameter mm')
        plt.subplot(212)
        plt.title('Right pupil')
        plt.scatter(x2,y2, s=0.1, c='grey')
        plt.xlim(0,x2[-1])
        plt.xlabel('time')
        plt.ylabel('pupil diameter mm')
        fig.show()
        
    def plot_variable(df,VAR, y_label=''):
        """
        Parameters
        ----------
        df : Pandas DataFrame
            Contains variables: 'time' und VAR.
        VAR : String
            Name of a variable in the DataFrame to be plot.
        y_label : String, optional
            Variable name. The default is VAR.

        Returns
        -------
        Plot of a variable over time
        """
        plt.plot(df['time'],df[VAR])
        plt.xlabel('time')
        if y_label == '':
            plt.ylabel(VAR)
        else:
            plt.ylabel(y_label)
        plt.show()    