import cv2
import numpy as np
from cluster_holo import fit_multisphere, bisphere, fit
import holopy as hp
from matplotlib import pyplot as plt


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

#check bisphere for guess first
holo, cluster = bisphere(a_p = a_p, n_p = n_p, z_p = zguess, theta = thetaguess, phi = phiguess)

#print(cluster)
#plt.imshow(holo)
#plt.show()

img = cv2.imread(imagepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float)
img /= np.mean(img)
fit(img, a_p=a_p*2, n_p=n_p, z_p=zguess, plot=True, percentpix=1)

result = fit_multisphere(imagepath, a_p = a_p, n_p = n_p, theta_guess = thetaguess, z_guess = zguess, phi_guess = phiguess)
