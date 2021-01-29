import holopy as hp
from holopy.scattering import Scatterer, Sphere, Spheres, calc_holo
from holopy.scattering.scatterer import Indicators
import numpy as np
import matplotlib
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import json
from pylorenzmie.theory import coordinates, LMHologram
from pylorenzmie.analysis.Feature import Feature
from holopy.scattering.theory import DDA
from scipy.spatial.transform import Rotation as R
from holopy.inference import prior, ExactModel
from holopy.core.process import normalize

#Instrument parameters:
wv = 0.447
mag = 0.120
n_m = 1.34 #assume water
px = 200

config = {'n_m': n_m, 'wavelength': wv, 'magnification': mag}

def feature_extent(a_p, n_p, z_p, config, nfringes=20, maxrange=300):
    '''Radius of holographic feature in pixels'''

    x = np.arange(0, maxrange)
    y = np.arange(0, maxrange)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    zv = np.zeros_like(xv)
    coordinates = np.stack((xv, yv, zv))

    h = LMHologram(coordinates=coordinates)
    h.instrument.properties = config
    h.particle.a_p = a_p
    h.particle.n_p = n_p
    h.particle.z_p = z_p
    # roughly estimate radii of zero crossings
    b = h.hologram() - 1.
    ndx = np.where(np.diff(np.sign(b)))[0] + 1
    if len(ndx) <= nfringes:
        return maxrange
    else:
        return float(ndx[nfringes])

def rotate(vector, axis, angle):
    rotation_vector = angle * axis
    rotation = R.from_rotvec(rotation_vector)
    rotated_vec = rotation.apply(vector)
    return rotated_vec

# cluster t-matrix version

def sphere(a_p, n_p, z_p):
    px = int(feature_extent(a_p, n_p, z_p, config))
    detector = hp.detector_grid(2*px, mag)
    center = (mag*px, mag*px, z_p)
    s = Sphere(center = center, n = n_p, r = a_p)
    
    holo = np.squeeze(calc_holo(detector, s, medium_index=n_m, illum_wavelen=wv, illum_polarization=(1, 0))).data
    
    #noise
    holo += np.random.normal(0., 0.05, holo.shape)
    
    return holo
    
    
def bisphere(a_p, n_p, z_p, theta, phi):
    px = int(feature_extent(a_p, n_p, z_p, config))*2
    detector = hp.detector_grid(2*px, mag)
    center = (mag*px, mag*px, z_p)
    
    delta = np.array([np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), np.sin(theta)])
    c1 =  + 1.001*a_p*delta
    c2 =  - 1.001*a_p*delta
    cluster = np.array([c1, c2])

    cluster_return = cluster.copy().tolist()

    cluster += center
    
    s1 = Sphere(center = cluster[0], n = n_p, r = a_p)
    s2 = Sphere(center = cluster[1], n = n_p, r = a_p)
    
    dimer = Spheres([s1, s2])
    
    holo = np.squeeze(calc_holo(detector, dimer, medium_index=n_m, illum_wavelen=wv, illum_polarization=(1, 0))).data
    
    #noise
    holo += np.random.normal(0., 0.05, holo.shape)
    
    return holo, cluster_return


def trisphere(a_p, n_p, z_p, alpha, theta, phi, check_geom=False):
    '''                                                                                                                                                                                                     
    alpha: angle of 3rd monomer wrt dimer                                                                                                                                                                   
    -alpha=pi/3: equilateral triangle                                                                                                                                                                       
    -alpha between pi/3 and 5pi/3 (no overlaps)                                                                                                                                                             
                                                                                                                                                                                                            
    theta, phi: rotation angles                                                                                                                                                                             
    -theta=0 : aligned along xy plane                                                                                                                                                                       
    -theta between 0 and 2pi                                                                                                                                                                                
    -phi between 0 and 2pi                                                                                                                                                                                  
    '''
    px = int(feature_extent(a_p, n_p, z_p, config))*2
    detector = hp.detector_grid(2*px, mag)
    center = (mag*px, mag*px, z_p)

    if alpha < np.pi/3 or alpha > 5*np.pi/3:
        raise Exception("Invalid value for alpha")

    #rotation about origin                                                                                                                                                                                  
    #delta = np.array([np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), np.sin(theta)])                                                                                                                
    delta = np.array([1,0,0])

    #angle of third monomer wrt dimer                                                                                                                                                                       
    delta_2 = np.array([-np.cos(alpha), np.sin(alpha), 0])
    c1 = 1.001*a_p*delta
    c2 = -1.001*a_p*delta
    c3 = c1 + 2.001*a_p*delta_2

    cluster = np.array([c1, c2, c3])

    #get centroid of trimer and re-center
    centroid = np.mean(np.array([c1, c2, c3]), axis=0)
    cluster -= centroid

    #rotate trimer
    zax = np.array([0,0,1])
    xax = np.array([1,0,0])
    zrot = lambda c : rotate(c, zax, phi)
    xrot = lambda c : rotate(c, xax, theta)

    cluster = zrot(cluster)
    cluster = xrot(cluster)

    cluster_return = cluster.copy().tolist()

    #place in particle position
    cluster += center

    s1 = Sphere(center = cluster[0], n = n_p, r = a_p)
    s2 = Sphere(center = cluster[1], n = n_p, r = a_p)
    s3 = Sphere(center = cluster[2], n = n_p, r = a_p)

    trimer = Spheres([s1, s2, s3])

    r_sum=4*a_p
    if check_geom:
        #geometry check                                                                                                                                                                                     
        npix=60 #npix x npix grid                                                                                                                                                                           
        coord_range = np.linspace(mag*px-r_sum, mag*px+r_sum, num=npix)
        x,y = np.meshgrid(coord_range, coord_range)
        trimer_points = np.zeros((npix, npix))
        for i in range(npix):
            for j in range(npix):
                coord = [x[i][j], y[i][j], z_p]
                if trimer.contains(coord):
                    trimer_points[i][j] = 1
        plt.imshow(trimer_points)
        plt.show()

    holo = np.squeeze(calc_holo(detector, trimer, medium_index=n_m, illum_wavelen=wv, illum_polarization=(1, 0))).data

    #noise                                                                                                                                                                                                  
    holo += np.random.normal(0., 0.05, holo.shape)

    return holo, cluster_return



def fit(data, a_p, n_p, z_p, plot=False, return_img=False, percentpix=0.1):
    feature = Feature(model=LMHologram())
    px = int(np.sqrt(data.size))
    
    ins = feature.model.instrument
    ins.wavelength = wv
    ins.magnification = mag
    ins.n_m = n_m
    
    feature.mask.distribution = 'fast'
    feature.mask.percentpix = percentpix

    x = np.arange(0, px)
    y = np.arange(0, px)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()
    zv = np.zeros_like(xv)
    coordinates = np.stack((xv, yv, zv))

    #feature.model.coordinates = coordinates((px, px), dtype=np.float32)
    feature.model.coordinates = coordinates
    feature.coordinates = coordinates
    p = feature.model.particle

    p.r_p = [px//2, px//2, z_p/mag]
    p.a_p = a_p
    p.n_p = n_p
    feature.data = np.array(data)
    #result = feature.optimize(method='lm', verbose=False)
    result = feature.optimize()
    print(result)
    if plot:
        plt.imshow(np.hstack([data, feature.hologram()]))
        plt.show()
    a_fit = feature.model.particle.a_p
    n_fit = feature.model.particle.n_p
    z_fit = feature.model.particle.z_p

    if return_img:
        return feature.model.hologram(), a_fit, n_fit, z_fit
    else:
        return a_fit, n_fit, z_fit




def fit_multisphere(data_path, a_p, n_p, z_guess, theta_guess, phi_guess, fit_a=False):
    
    px = cv2.imread(data_path).shape[0]

    data_holo = hp.load_image(data_path, spacing = mag, medium_index = n_m,
                              illum_wavelen = wv, illum_polarization = (1,0), channel=0)
 
    data_holo = normalize(data_holo)
    z_p = prior.Uniform(lower_bound=45, upper_bound=100, guess=z_guess, name='z_p')
    theta = prior.Uniform(lower_bound=0, upper_bound=np.pi/2, guess = theta_guess, name='theta')
    phi = prior.Uniform(lower_bound=0, upper_bound=np.pi, guess=phi_guess, name='phi')
    
    '''
    #idk why this doesn't work, maybe the fitter has some issue with numpy
    center = (mag*px, mag*px, z_p)
    delta = np.array([np.cos(phi)*np.cos(theta), np.sin(phi)*np.cos(theta), np.sin(theta)])
    c1 =  + 1.001*a_p*delta
    c2 =  - 1.001*a_p*delta
    cluster = np.array([c1, c2])

    s1 = Sphere(center = cluster[0], n = n_p, r = a_p)
    s2 = Sphere(center = cluster[1], n = n_p, r = a_p)
    '''

    if fit_a:
        a_1 = prior.Uniform(lower_bound=a_p*0.8, upper_bound = a_p*1.2, guess=a_p, name='a_1')
        a_2 = prior.Uniform(lower_bound=a_p*0.8, upper_bound = a_p*1.2, guess=a_p, name='a_2')
    else:
        a_1 = a_p
        a_2 = a_p
    
    x1 = mag*px/2 + a_1*np.cos(phi)*np.cos(theta)*1.001
    x2 = mag*px/2 - a_2*np.cos(phi)*np.cos(theta)*1.001
    y1 = mag*px/2 + a_1*np.cos(theta)*np.sin(phi)*1.001
    y2 = mag*px/2 + a_2*np.cos(theta)*np.sin(phi)*1.001
    z1 = z_p + a_1*np.sin(theta)*1.001
    z2 = z_p - a_2*np.sin(theta)*1.001
    
    s1 = Sphere(center = [x1, y1, z1], n = n_p, r = a_1)
    s2 = Sphere(center = [x2, y2, z2], n = n_p, r = a_2)
    

    dimer = Spheres([s1, s2], warn=False)

    model = ExactModel(scatterer=dimer, calc_func=calc_holo,
                   noise_sd = None, medium_index = n_m, illum_wavelen=wv, illum_polarization=(1,0))
    fit_result = hp.fit(data_holo, model)

    return fit_result


