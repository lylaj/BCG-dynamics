# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 17:22:34 2017

@author: bluegreen973
"""

import scipy.optimize as so
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

def plot(data_x, data_y, nbins=30, ranges=[], smooth=1, color='gray', label = False, linestyle = '-',linewidth = 2,manual_locations = False, sig05 = True,sig1 = True, sig2 = False,clabel = True, only_level = False):
	def find_confidence_interval(x, pdf, confidence_level):
		return pdf[pdf > x].sum() - confidence_level
	
	kde = ss.gaussian_kde(np.vstack((data_x,data_y)))
	
	if ranges == []:
		ranges = [[np.min(data_x),np.max(data_x)],[np.min(data_y),np.max(data_y)]]
		
	x = np.linspace(ranges[0][0],ranges[0][1], nbins)
	y = np.linspace(ranges[1][0],ranges[1][1], nbins)
	'''
	z, x, y = np.histogram2d(data_x, data_y, bins = nbins, normed=True, range =ranges )
	hist = ndimage.gaussian_filter(z, sigma=smooth, order=0)
	'''
	grid = np.hstack((np.meshgrid(x,y)[0].reshape(len(x)*len(y),1), np.meshgrid(x,y)[1].reshape(len(x)*len(y),1)))
	
	hist = kde(grid.T).reshape(len(x),len(y))
	x_bin_sizes = (x[1:] - x[:-1]).reshape((1,nbins-1))
	y_bin_sizes = (y[1:] - y[:-1]).reshape((nbins-1,1))
	
	pdf = (hist[:-1,:-1]*(x_bin_sizes*y_bin_sizes))
	
	levels = []
	strs = []
	if sig2 ==True:
		two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.95))
		levels.append(two_sigma)
		strs.append(' 2 $\sigma$ ')
	if sig1 ==True:
		one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
		levels.append(one_sigma)
		strs.append(' 1 $\sigma$ ')
	if sig05 ==True:
		half_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.38))
		levels.append(half_sigma)
		strs.append(' 0.5 $\sigma$ ')

	if only_level == False:
		if sig1==True and sig05 ==True and sig2 == False:
			CS = plt.contour((x[1:]+x[:-1])/2,(y[1:]+y[:-1])/2,pdf,levels = [levels[0]], linestyles = 'dashed', colors = color, label = label, linewidths = linewidth)
			CS = plt.contour((x[1:]+x[:-1])/2,(y[1:]+y[:-1])/2,pdf,levels = [levels[1]], linestyles = '-', colors = color, label = label, linewidths = linewidth)
		else:
			CS = plt.contour((x[1:]+x[:-1])/2,(y[1:]+y[:-1])/2,pdf,levels = levels, linestyles = linestyle, colors = color, label = label, linewidths = linewidth)
			
			
	elif only_level == True:
		return(levels)
	'''
	if clabel == True:
		fmt = {}
		for l,s in zip( CS.levels, strs ):
			fmt[l] = s	
			plt.clabel(CS,CS.levels,inline=True,fmt=fmt,fontsize=12, manual=manual_locations)
	'''
	