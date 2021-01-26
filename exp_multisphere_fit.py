import cv2
import numpy as np
from cluster_holo import fit_multisphere, bisphere, fit
import holopy as hp
from matplotlib import pyplot as plt
import json
from time import time


#Instrument parameters:                                                                                
wv = 0.447
mag = 0.120
n_m = 1.34 #assume water                                                                               
config = {'n_m': n_m, 'wavelength': wv, 'magnification': mag} 

n_p = 1.435
a_p = 0.527

image_fnames = ['crop01_square.png', 'crop02.png', 'crop03.png', 'crop04.png']
imagebase = 'data/exp_crops/'

#these initial guesses seem to not error out so i'm keeping them for now
zguess = 70
thetaguess = np.pi/2*0.8
phiguess = 0.1

savedir = {'a_p': a_p, 'n_p': n_p, 'imagename': [], 'sphere_a_fit': [], 'sphere_n_fit':[], 'sphere_z_fit': [],
           'spherefit_time': [],
           'multisphere_theta_fit': [], 'multisphere_phi_fit':[], 'multisphere_z_fit':[],
           #'multisphere_redchi': [],
           'multispherefit_time':[]}
for fname in image_fnames:
    ##check bisphere for guess first
    #holo, cluster = bisphere(a_p = a_p, n_p = n_p, z_p = zguess, theta = thetaguess, phi = phiguess)
    #print(cluster)
    #plt.imshow(holo)
    #plt.show()
    print(fname)
    savedir['imagename'].append(fname)

    
    imagepath = imagebase + fname
    img = cv2.imread(imagepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float)
    img /= np.mean(img)
    start = time()
    a,z,n = fit(img, a_p=a_p*2, n_p=n_p, z_p=zguess, percentpix=1)
    delta = time() - start

    savedir['sphere_a_fit'].append(a)
    savedir['sphere_n_fit'].append(n)
    savedir['sphere_z_fit'].append(z)
    savedir['spherefit_time'].append(delta)

    start = time()
    result = fit_multisphere(imagepath, a_p = a_p, n_p = n_p, theta_guess = thetaguess, z_guess = zguess, phi_guess = phiguess)
    delta = time() - start
    best_fit_values = result.parameters


    saveholo = np.clip(np.array(result.hologram)*100., 1, 255)
    cv2.imwrite(imagebase + 'fits/'+ fname, saveholo)
    print(best_fit_values)
    
    savedir['multisphere_theta_fit'].append(best_fit_values['theta'])
    savedir['multisphere_phi_fit'].append(best_fit_values['phi'])
    savedir['multisphere_z_fit'].append(best_fit_values['z_p'])
    #savedir['multisphere_redchi'].append(result.chisq)
    savedir['multispherefit_time'].append(delta)

with open('data/exp_multisphere_fits.json', 'w') as f:
    json.dump(savedir, f)
