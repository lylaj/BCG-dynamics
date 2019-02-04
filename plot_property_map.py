# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:11:58 2017

@author: bluegreen973
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.stats as ss
#import subsubplot
#plt.style.use('seoyoung')


def plot(data_x, data_y, data_z, npix, bin_type = 'avg', vmax = False, vmin = False, cmap = 'jet', smooth = 0,log = True, data_m=[], ranges = False, cbar=False,cbar_orientation = 'vertical',cbar_inside = False, cbar_title = False, binlabel = False,fontcolor = 'white',num_crit = 1, contour = False, return_index = False, return_err = False):

	print("START PLOT")
	if ranges == False :
		x_start = np.min(data_x)
		y_start = np.min(data_y)		
		x_end = np.max(data_x)
		y_end = np.max(data_y)

	else :
		x_start = ranges[0][0]
		y_start = ranges[1][0]
		x_end = ranges[0][1]
		y_end = ranges[1][1]

	x = np.linspace(x_start,x_end, npix)
	y = np.linspace(y_start,y_end, npix)	
	
	z = np.zeros((npix-1,npix-1))
	err = np.zeros((npix-1,npix-1))
	nn = np.zeros((npix-1,npix-1))
	
	
	nn = ss.binned_statistic_2d(data_x,data_y, None, 'count', bins=[x,y]).statistic

	def avg_val_err(z):
		return(np.std(z)/len(z)**0.5)
	def med_val_err(z):
		return(1.2533*np.std(z)/len(z)**0.5)
	def frac_err_dum(z):
		return(((1-z)*z)**0.5)
		
	if bin_type  == 'avg':
		z = ss.binned_statistic_2d(data_x,data_y, values = data_z, statistic = np.mean, bins = [x,y]).statistic
		if return_err == 'val':
			err = ss.binned_statistic_2d(data_x,data_y, values = data_z, statistic = avg_val_err, bins = [x,y]).statistic
		elif return_err == 'frac':
			dum = ss.binned_statistic_2d(data_x,data_y, values = data_z, statistic = frac_err_dum, bins = [x,y]).statistic
			err = dum/nn**0.5
			
	elif bin_type  == 'median':
		z = ss.binned_statistic_2d(data_x,data_y, values = data_z, statistic = np.median, bins = [x,y]).statistic
		if return_err == 'val':
			err = ss.binned_statistic_2d(data_x,data_y, values = data_z, statistic = med_val_err, bins = [x,y]).statistic
		elif return_err == 'frac':
			dum = ss.binned_statistic_2d(data_x,data_y, values = data_z, statistic = frac_err_dum, bins = [x,y]).statistic
			err = dum/nn**0.5

	z[nn<=num_crit] = 'nan'
	#z = np.rot90(z[::-1])

	err[nn<=num_crit] = 'nan'
	#err = np.rot90(err[::-1])
		
		
	'''
	for i in range(0,npix-1):
		for j in range(0,npix-1):
			data_cell_index = np.where( (data_x>x[i])*(data_x<=x[i+1])*(data_y>y[j])*(data_y<=y[j+1]) )
						
			nn[j,i]=len(data_cell_index[0])
			if nn[j,i]>num_crit:
				if len(data_m)==0:
					if bin_type  == 'avg':
						z[j,i] = np.sum(data_z[data_cell_index])/len(data_z[data_cell_index])
					elif bin_type  == 'median':
						z[j,i] = np.median(data_z[data_cell_index])

					if return_err =='val':
						err[j,i] = np.std(data_z[data_cell_index])/len(data_z[data_cell_index])**0.5
						if bin_type =='median':
							err[j,i] = err[j,i]*1.2533
					elif return_err == 'frac':
						err[j,i] =  ((1-z[j,i])*z[j,i]/nn[j,i])**0.5
					
				else:
					if bin_type == 'avg':
						z[j,i] = np.sum(data_z[data_cell_index]*data_m[data_cell_index]) / np.sum(data_m[data_cell_index])
					#err[j,i] = np.str(data_z[data_cell_index])*len(data_cell_index[0])**-0.5

			else :
				#z[j,i] = 'nan'
				data_wide_index = np.where( (data_x>x[i]-(x_end-x_start)/(npix-1)/2)*(data_x<=x[i+1]+(x_end-x_start)/(npix-1)/2)*(data_y>y[j]-(y_end-y_start)/(npix-1)/2)*(data_y<=y[j+1])+(y_end-y_start)/(npix-1)/2 )

				if bin_type  == 'avg':
					z[j,i] = np.sum(data_z[data_wide_index])/len(data_z[data_wide_index])
				elif bin_type  == 'median':
					z[j,i] = np.median(data_z[data_wide_index])	
	'''
	if smooth >0:
		z = ndimage.filters.gaussian_filter(z,smooth)
		#z = ndimage.filters.convolve(z,weights = nn)
		z[nn<=num_crit]='nan'
	else:
		z[nn<=num_crit]='nan'

				
	if binlabel == 'percent':
		for i in range(0,npix-1):
			for j in range(0,npix-1):
				try:
					plt.text(x_start+(i+0.5)*(x_end-x_start)/(npix-1), y_start+(j+0.5)*(y_end-y_start)/(npix-1), str(int(z.T[i,j]*100))+"%", horizontalalignment = 'center', verticalalignment = 'center', fontsize = 11, color = fontcolor, fontweight='bold')
				except: pass
	elif binlabel == 'avg':
		for i in range(0,npix-1):
			for j in range(0,npix-1):
				if nn.T[i,j]>num_crit:
					plt.text(x_start+(i+0.5)*(x_end-x_start)/(npix-1), y_start+(j+0.5)*(y_end-y_start)/(npix-1), str(round(np.mean(z.T[i,j]),2)), horizontalalignment = 'center', verticalalignment = 'center', fontsize = 11, color = fontcolor, fontweight='bold')

	elif binlabel == 'num':			
		for i in range(0,npix-1):
			for j in range(0,npix-1):
				if nn.T[i,j]>num_crit:
					plt.text(x_start+(i+0.5)*(x_end-x_start)/(npix-1), y_start+(j+0.5)*(y_end-y_start)/(npix-1), "n = "+str(int(nn[j,i])), horizontalalignment = 'center', verticalalignment = 'center', fontsize = 9, color = fontcolor)			
	if contour ==True:
		plt.contour(z, colors='white', extent = [x_start,x_end,y_start,y_end], aspect='auto',smooth=1, levels = [np.percentile(z[nn>num_crit],50),np.percentile(z[nn>num_crit],60),np.percentile(z[nn>num_crit],70),np.percentile(z[nn>num_crit],80),np.percentile(z[nn>num_crit],90)] )
		

	if vmax == False:
		cax = plt.imshow(z.T[::-1], cmap = cmap, extent = [x_start,x_end,y_start,y_end], aspect='auto')
	else:
		cax = plt.imshow(z.T[::-1], vmax = vmax, vmin =vmin,interpolation= 'nearest', cmap = cmap, extent = [x_start,x_end,y_start,y_end],aspect='auto')
	if cbar == True and cbar_inside == False:
		cbar = plt.colorbar(shrink=0.8, orientation = cbar_orientation)
		if cbar_title != False:
			cbar.set_label(cbar_title, fontsize = 10)
	elif cbar == True and cbar_inside == True:
		
		#cbaxes = subsubplot.figure(plt.gca(),[0.7, 0.85, 0.25, 0.06])
		cbar = plt.colorbar(cax,cax=cbaxes,ticks = np.linspace(vmin,vmax,3), orientation='horizontal')
		#plt.axis('off')

		if cbar_title != False:
			cbar.set_label(cbar_title, fontsize = 10)
			
			
	if return_index == True:
		if return_err != False:
			return(z,x_start,x_end,y_start,y_end,nn,err)

		else:
			return(z,x_start,x_end,y_start,y_end,nn)
	#plt.xlabel("log density")
	#plt.ylabel("log Temperature")

	#ax = plt.gca()
	#ax.set_xticks(np.linspace(0,npix,5))
	#xticks = ["{:.2f}".format(x) for x in np.linspace(x_start,x_end, num=5)]
	#ax.set_xticklabels(xticks)
	#ax.set_yticks(np.linspace(0,npix,6))
	#yticks = ["{:.2f}".format(y) for y in np.linspace(y_end, y_start, num=6)]
	#ax.set_yticklabels(yticks)	

	print("END PLOT")

	
	
	
	
	
	
