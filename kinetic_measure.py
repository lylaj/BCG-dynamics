#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.optimize as so
import scipy.stats as ss
import pynbody
import matplotlib.pyplot as plt
import os.path
from numpy.linalg import eig, inv

###############################
####### 3D measurements #######
###############################

def dot(a,b):
    if a.shape==(3,): a = a.reshape(1,3)
    if b.shape==(3,): b = b.reshape(1,3)
    return(a[:,0]*b[:,0]+a[:,1]*b[:,1]+a[:,2]*b[:,2])

def get_spin_axis(part):
    ang_mom_vec = pynbody.analysis.angmom.ang_mom_vec(part) #Msol kpc km s**-1

    ang_mom = np.linalg.norm(ang_mom_vec)
    spin_axis = ang_mom_vec/ang_mom        
    return(spin_axis)

def get_r_axis(part, spin_axis = []):
    x_vec = part['pos']
    v_vec = part['vel']
    
    if spin_axis == []:
        spin_axis = get_spin_axis(part)
        
    r_axis = x_vec.view(type=np.ndarray)-dot(x_vec,spin_axis).reshape(len(x_vec),1)*spin_axis.reshape(1,3)
    r_axis = r_axis.view(type=np.ndarray)
    r_distance = np.linalg.norm(r_axis,axis=1)
    r_axis = r_axis/r_distance.reshape(len(r_distance),1)   
    return(r_axis)

def get_tan_axis(part, spin_axis = [], r_axis = []):
    if spin_axis == []:
        spin_axis = get_spin_axis(part)  
    if r_axis == []:
        spin_axis = get_r_axis(part)
        
    tan_axis = np.cross(spin_axis,r_axis)
    return(tan_axis)

def get_v_rot(part, spin_axis = [], r_axis = [], tan_axis = []):
    if tan_axis == []:
        if spin_axis == []:
            spin_axis = get_tan_axis(part)
        if r_axis == []:
            r_axis = get_r_axis(part, spin_axis)
            
        tan_axis = get_tan_axis(part, spin_axis, r_axis) 
   
    v_rot = dot(part['vel'], tan_axis)
    return(v_rot)

def kine_3d(part, spin_axis = [],profile = False, spin_part = []):
    if spin_axis == []:
        if profile == True:
            spin_axis = get_spin_axis(part)
        else:
            spin_axis = get_spin_axis(spin_part)
       
    r_axis = get_r_axis(part, spin_axis = spin_axis)
    
    tan_axis = get_tan_axis(part, spin_axis=spin_axis, r_axis=r_axis)

    v_rot_mean = np.abs(np.mean(get_v_rot(part, spin_axis, r_axis, tan_axis)))

    sig_r = np.std(dot(part['vel'], r_axis))
    sig_theta = np.std(dot(part['vel'], tan_axis))
    sig_z = np.std(dot(part['vel'], spin_axis))

    sig_3d = np.sqrt(sig_r**2+sig_theta**2+sig_z**2)
    return(v_rot_mean,sig_3d)






###############################
####### IFS observation #######
###############################

def gaussian(x, height, center, width):
    return height*np.exp(-(x - center)**2/(2*width**2))

def err(x, t, y):
	return np.sqrt(np.sum((gaussian(t,x[0],x[1],x[2])-y)**2))

def gauss_hermite(x, height, center, width,h3,h4):

    def H3_func(y):
        return((2**1.5*y**3-3*2**0.5*y)/6**0.5)
    def H4_func(y):
        return((4*y**4-12*y**2+3)/24**0.5)	

    return(height*np.exp(-(x - center)**2/(2*width**2))*(1+h3*H3_func((x - center)/width)+h4*H4_func((x - center)/width)))




def err_gh(x, t, height, center, width, y):
	return np.sqrt(np.sum((gauss_hermite(t,height, center, width,x[0],x[1])-y)**2))

	

out_dir = '/home/seoyoung/analysis/'


def get_rgb(s_filt, filt_size, nbin, add_info = [], re_cal = False, save = True):
	if (re_cal==False) and (os.path.isfile(out_dir +"render_vband_"+str(filt_size*2)+"kpc"+str(nbin)+"bins"+add_info+".csv") == True):

		vband = np.loadtxt(out_dir +"render_vband_"+str(filt_size*2)+"kpc"+str(nbin)+"bins"+add_info+".csv", delimiter=",")
		
	else:

		rgb = pynbody.plot.stars.render(s_filt, width = str(2*filt_size)+' kpc', resolution = nbin, dynamic_range = 4,plot = False,ret_im = True)
		
		vband = rgb[:,:,1]
		if save == True:
		    np.savetxt(out_dir +"render_vband_"+str(filt_size*2)+"kpc"+str(nbin)+"bins"+add_info+".csv",vband, delimiter=",")
						
	return(vband)


	
def get_IFS(part, half_size = 50,lbin = 1.5, nbin = False, add_info = [], re_cal = False, save = True):
	
	def fit_gaussian_v(x, success = False):
	    his_data = np.histogram(x, bins = 30)
	    optim = so.minimize(err,x0 = (1,0,200), args=((his_data[1][1:]+his_data[1][:-1])/2, his_data[0]), bounds = so.Bounds([0,-np.inf,0.1],[np.inf,np.inf,np.inf]))
	    #print(optim['success'])
	    return(optim['x'][1])
					
	def fit_gaussian_sig(x, success = False):
	    his_data = np.histogram(x, bins = 30)
	    optim = so.minimize(err,x0 = (1,0,200), args=((his_data[1][1:]+his_data[1][:-1])/2, his_data[0]), bounds = so.Bounds([0,-np.inf,0.1],[np.inf,np.inf,np.inf]))
	    #print(optim['success'])
	    return(optim['x'][2])

	def fit_gaussian_success(x, success = False):
	    his_data = np.histogram(x, bins = 30)
	    optim = so.minimize(err,x0 = (1,0,200), args=((his_data[1][1:]+his_data[1][:-1])/2, his_data[0]), bounds = so.Bounds([0,-np.inf,0.1],[np.inf,np.inf,np.inf]))
	    #print(optim['success'])
	    return(optim['success'])					
		
	def fit_GH_h3(x, success = False):
	    his_data = np.histogram(x, bins = 30)
	    optim = so.minimize(err,x0 = (1,0,200), args=((his_data[1][1:]+his_data[1][:-1])/2, his_data[0]), bounds = so.Bounds([0,-np.inf,0.1],[np.inf,np.inf,np.inf]))
	
	    optim_gh = so.minimize(err_gh,x0 = (0,0), args=((his_data[1][1:]+his_data[1][:-1])/2, his_data[0], optim['x'][0], optim['x'][1], optim['x'][2]), bounds = so.Bounds([-0.5,-0.5],[0.5,0.5]))	    #print(optim['success'])
	    return(optim_gh['x'][0])

	def fit_GH_h4(x, success = False):
	    his_data = np.histogram(x, bins = 30)
	    optim = so.minimize(err,x0 = (1,0,200), args=((his_data[1][1:]+his_data[1][:-1])/2, his_data[0]), bounds = so.Bounds([0,-np.inf,0.1],[np.inf,np.inf,np.inf]))
	
	    optim_gh = so.minimize(err_gh,x0 = (0,0), args=((his_data[1][1:]+his_data[1][:-1])/2, his_data[0], optim['x'][0], optim['x'][1], optim['x'][2]), bounds = so.Bounds([-0.5,-0.5],[0.5,0.5]))	    #print(optim['success'])
	    return(optim_gh['x'][1])

				
	
	if (re_cal==False) and (os.path.isfile(out_dir +"IFU_v_star_"+str(half_size*2)+"kpc"+str(nbin)+"bins"+add_info+".csv") == True):
		v_spax = np.loadtxt(out_dir +"IFU_v_star_"+str(half_size*2)+"kpc"+str(nbin)+"bins"+add_info+".csv", delimiter=",")
		sig_spax = np.loadtxt(out_dir +"IFU_sig_star_"+str(half_size*2)+"kpc"+str(nbin)+"bins"+add_info+".csv", delimiter=",")
		#h3 = np.loadtxt(out_dir +"IFU_h3_star_"+str(half_size*2)+"kpc"+str(nbin)+"bins"+add_info+".csv", delimiter=",")
		#h4 = np.loadtxt(out_dir +"IFU_h4_star_"+str(half_size*2)+"kpc"+str(nbin)+"bins"+add_info+".csv", delimiter=",")
		
	else:
	    #part['pos'] = part['pos'].in_units('kpc h^-1')

	    print("caluculate binned statistic")
	    pynbody.analysis.halo.vel_center(part)
	    if nbin == False:
	        grid = np.arange(-half_size,half_size,lbin)
	    else:
	        grid = np.linspace(-half_size,half_size,nbin+1)

	    count_stat = ss.binned_statistic_2d(part['x'],part['y'], None, 'count', bins=[grid,grid]).statistic

	    success_fit = ss.binned_statistic_2d(part['x'],part['y'], values = part['vel'][:,2], statistic = fit_gaussian_success, bins = [grid,grid]).statistic
		
		
	    #v_spax = ss.binned_statistic_2d(part['x'],part['y'], values = part['vel'][:,2], statistic = fit_gaussian_v, bins = [grid,grid]).statistic
	    v_spax = ss.binned_statistic_2d(part['x'],part['y'], values = part['vel'][:,2], statistic = np.mean, bins = [grid,grid]).statistic
	    v_spax[(count_stat<10)+(success_fit==False)] = 'nan'
					
	    sig_spax = ss.binned_statistic_2d(part['x'],part['y'], values = part['vel'][:,2], statistic = np.std, bins = [grid,grid]).statistic
	    sig_spax[(count_stat<10)+(success_fit==False)] = 'nan'


	    def moment_3(array):
	    	return(ss.moment(array,moment = 3))
	    def moment_4(array):
	    	return(ss.moment(array,moment = 4))

	    h3 = ss.binned_statistic_2d(part['x'],part['y'], values = part['vel'][:,2], statistic = moment_3, bins = [grid,grid]).statistic
	    h3[(count_stat<10)+(success_fit==False)] = 'nan'
					
	    h4 = ss.binned_statistic_2d(part['x'],part['y'], values = part['vel'][:,2], statistic = moment_4, bins = [grid,grid]).statistic
	    h4[(count_stat<10)+(success_fit==False)] = 'nan'
					

	    if save == True:
		    np.savetxt(out_dir +"IFU_v_star_"+str(half_size*2)+"kpc"+str(nbin)+"bins"+add_info+".csv",v_spax, delimiter=",")
		    np.savetxt(out_dir +"IFU_sig_star_"+str(half_size*2)+"kpc"+str(nbin)+"bins"+add_info+".csv",sig_spax, delimiter=",")
	
		    np.savetxt(out_dir +"IFU_h3_star_"+str(half_size*2)+"kpc"+str(nbin)+"bins"+add_info+".csv",h3, delimiter=",")
		    np.savetxt(out_dir +"IFU_h4_star_"+str(half_size*2)+"kpc"+str(nbin)+"bins"+add_info+".csv",h4, delimiter=",")

	return(v_spax, sig_spax)
	#return(v_spax, sig_spax,h3,h4)

def get_IFS_kine(part, nbin, half_size, v_spax = [], sig_spax = [], rgb_weight = []):
	
	if rgb_weight == []:
		rgb_weight = pynbody.plot.stars.render(part, width = str(half_size*2)+' kpc', resolution = nbin, dynamic_range = 4,ret_im = True)
	
	if v_spax  == [] or sig_spax == []:
		v_spax,sig_spax = get_IFS(part, half_size, nbin)
		
	v_spax2 = v_spax**2
	v_spax2=v_spax2.reshape(v_spax.shape)
	v_spax2 = v_spax2[v_spax>-999]
	
	sig_spax2 = sig_spax**2
	sig_spax2=sig_spax2.reshape(sig_spax.shape)
	sig_spax2 = sig_spax2[sig_spax>-999]
	
	sig_ifs = np.sum(10**rgb_weight[:,:,1][v_spax>-999].T*sig_spax[sig_spax>-999])/np.sum(10**rgb_weight[:,:,1][sig_spax>-999].T)
	vos_ifs = np.sqrt(np.sum(10**rgb_weight[:,:,1][v_spax>-999].T*v_spax2)/np.sum(10**rgb_weight[:,:,1][sig_spax>-999].T*sig_spax2))
	
	return(sig_ifs, vos_ifs)
	
def get_IFS_image(s_filt,i):
	rgb = get_rgb(s_filt, filt_size/s_filt.properties['h'], nbin,add_info = "_random"+str(i)+"_z_"+zred, re_cal = re_cal)

	spin_axis_tot = get_spin_axis(s_filt.star)

	print("get IFS data")
	v_spax, sig_spax = get_IFS(s_filt.star,half_size = filt_size/s_filt.properties['h'], nbin = nbin, add_info = "_random"+str(i)+"_z_"+zred, re_cal = re_cal)

	plt.subplot(3,4,i*4-3)
	print("get formation time plot")
	ppm.plot(s_filt.star['pos'][:,0],s_filt.star['pos'][:,1], s_filt.star['tform'], npix = nbin, vmax = 6.5,vmin = 2, cmap = 'jet', log = False, cbar = True)
	plt.contour(rgb, colors = 'k', extent = [-filt_size/s_filt.properties['h'],filt_size/s_filt.properties['h'],-filt_size/s_filt.properties['h'],filt_size/s_filt.properties['h']])

	plt.subplot(3,4,i*4-2)
	print("get metalicity plot")
	ppm.plot(s_filt.star['pos'][:,0],s_filt.star['pos'][:,1], s_filt.star['feh'], vmax = 0.01, vmin = -1,npix = nbin, cmap = 'jet', log = False, cbar = True)
	plt.contour(rgb, colors = 'k', extent = [-filt_size/s_filt.properties['h'],filt_size/s_filt.properties['h'],-filt_size/s_filt.properties['h'],filt_size/s_filt.properties['h']])


	plt.subplot(3,4,i*4-1)
	plt.imshow(v_spax.T[::-1], cmap = 'jet', vmax = 500, vmin = -500,  origin='upper', extent = [-filt_size/s_filt.properties['h'],filt_size/s_filt.properties['h']+2*filt_size/s_filt.properties['h']/nbin,-filt_size/s_filt.properties['h'],filt_size/s_filt.properties['h']+2*filt_size/s_filt.properties['h']/nbin])
	plt.colorbar()

	plt.contour(rgb, colors = 'k', extent = [-filt_size/s_filt.properties['h'],filt_size/s_filt.properties['h'],-filt_size/s_filt.properties['h'],filt_size/s_filt.properties['h']])

	plt.subplot(3,4,4*i)
	plt.imshow(sig_spax.T[::-1], cmap = 'jet',vmax = 500, vmin = 0,  origin='upper', extent = [-filt_size/s_filt.properties['h'],filt_size/s_filt.properties['h']+2*filt_size/s_filt.properties['h']/nbin,-filt_size/s_filt.properties['h'],filt_size/s_filt.properties['h']+2*filt_size/s_filt.properties['h']/nbin])
	plt.colorbar()

	plt.contour(rgb, colors = 'k', extent = [-filt_size/s_filt.properties['h'],filt_size/s_filt.properties['h'],-filt_size/s_filt.properties['h'],filt_size/s_filt.properties['h']])





    	
def bin_edges_equalN(x, nbin):
	npt = len(x)
	return np.interp(np.linspace(0, npt, nbin + 1),np.arange(npt),np.sort(x))

def get_slit(sim, half_size, nbin, align = [],aparture = 11):

	part = sim.star
	a,b,phi, center, ellip = isophote_fit(part)

	spin_axis = get_spin_axis(part[part['r']<15])
	theta_spin = np.pi/2-np.arctan(spin_axis[1]/spin_axis[0])

	if align == 'spin':
		#spin_axis = get_spin_axis(s.star[s.star['r']<10])
		theta = theta_spin

	elif align == 'phot':
		theta = -phi


	sim.rotate_z(theta*180/np.pi)

	s_filt = sim[pynbody.filt.Disc(str(aparture)+' kpc h^-1', '35 kpc h^-1')]

	part = s_filt.star[(np.abs(s_filt.star['y'])<1)*(np.abs(s_filt.star['x'])<half_size)]

	slit_bins = bin_edges_equalN(part['x'],nbin)
	
	v_spax = ss.binned_statistic(part['x'],part['vel'][:,2], statistic=np.mean,bins = slit_bins).statistic
	sig_spax = ss.binned_statistic(part['x'],part['vel'][:,2], statistic=np.std,bins = slit_bins).statistic

	r_spax = (slit_bins[:-1]+slit_bins[1:])/2

	sig_0 = np.std(s_filt.star['vel'][np.where(s_filt.star['rxy']<5),2])

	sim.rotate_z(-theta*180/np.pi)


	return(v_spax,sig_spax, r_spax,sig_0, ellip, theta - theta_spin)


def get_beta(part,slit_bins):
	axis_r = part['pos']/part['r'].reshape((len(part),1))
	axis_theta = np.cross(axis_r, np.array((0,0,1)))
	axis_theta = axis_theta/np.linalg.norm(axis_theta, axis = 1).reshape((len(part),1))
	axis_phi = np.cross(axis_r,axis_theta)

	v_r = dot(part['vel'],axis_r)
	v_theta = dot(part['vel'],axis_theta)
	v_phi = dot(part['vel'],axis_phi)


	sig_r = ss.binned_statistic(part['x'],v_r, statistic=np.std,bins = slit_bins).statistic
	sig_theta = ss.binned_statistic(part['x'],v_theta, statistic=np.std,bins = slit_bins).statistic
	sig_phi = ss.binned_statistic(part['x'],v_phi, statistic=np.std,bins = slit_bins).statistic

	return(1-(sig_theta**2+sig_phi**2)/(2*sig_r**2))


###############################
####### Ellipse fitting #######
###############################
# Reference : http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
#https://en.wikipedia.org/wiki/Ellipse


def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])



def ellipse_angle_of_rotation2( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

def ellipse_angle_of_rotation_test(a):

	b,c,d,f,g,a = a[1],a[2],a[3],a[4],a[5],a[0]
	if b == 0:
		if a < c:
			return 0
		else:
			return np.pi/2	
	else:
		return np.arctan((c-a-np.sqrt((a-c)**2+b**2))/b)

def isophote_fit(part,width = '40 kpc h^-1'):


	image = pynbody.plot.sph.image(part, qty='k_lum_den', units = 'pc^-2', width=width, log=False, clear=False, noplot=False, resolution = 100)

	cs = plt.contour(pynbody.plot.stars.convert_to_mag_arcsec2(image), levels = [20], extent = [-30,30,-30,30], colors = 'b')
	p = cs.allsegs

	cs_len = 0
	for i in range(0,len(p[0])):
		if cs_len<len(p[0][i]):
			cs_len = len(p[0][i])
			cs_i = i

	a = fitEllipse(p[0][cs_i][:,0],p[0][cs_i][:,1])
	if a[0]<0: a = -a
	center = ellipse_center(a)
	#phi = ellipse_angle_of_rotation(a)
	phi = ellipse_angle_of_rotation_test(a)
	axes = ellipse_axis_length(a)

	# get the individual axes
	a, b = axes


	if a<0 or b<0:
		ellip = np.nan
	else:
		if a>b:
			ellip = (1-(b/a)**2)**0.5
		else:
			ellip = (1-(a/b)**2)**0.5


	return(a,b,phi, center, ellip)










	

