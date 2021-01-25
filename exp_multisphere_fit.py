import cv2
import numpy as np
from cluster_holo import fit_multisphere
import holopy as hp


#Instrument parameters:                                                                                
wv = 0.447
mag = 0.120
n_m = 1.34 #assume water                                                                               
config = {'n_m': n_m, 'wavelength': wv, 'magnification': mag} 

n_p = 1.435
a_p = 0.527


imagepath = 'data/exp_crops/crop02.png'


zguess = 70
thetaguess = np.pi/2*0.8
phiguess = 0

result = fit_multisphere(imagepath, a_p = a_p, n_p = n_p, theta_guess = thetaguess, z_guess = zguess, phi_guess = phiguess)
