import cv2
import numpy as np
from cluster_holo import fit_multisphere, bisphere, fit
import holopy as hp
from matplotlib import pyplot as plt
import json
from time import time
#import pickle


#Instrument parameters:                                                                                
wv = 0.447
mag = 0.120
n_m = 1.34 #assume water                                                                               
config = {'n_m': n_m, 'wavelength': wv, 'magnification': mag}

with open('data/exp_fits_bootstrap01.json', 'r') as f:
    bd = json.load(f)

n_p = 1.435
a_p = 0.527

image_fnames = ['crop01_square.png', 'crop02.png', 'crop03.png', 'crop04.png']
imagebase = 'data/exp_crops/'

#these initial guesses seem to not error out so i'm keeping them for now
#zguess = 70
#thetaguess = np.pi/2*0.8
phiguesses = [0.3, np.pi/2, 0.5, 0.1]

savedir = {'a_p': [], 'n_p': [], 'imagename': [], 'sphere_a_fit': [], 'sphere_n_fit':[], 'sphere_z_fit': [],
           'spherefit_time': [],
           'theta_fit': [], 'theta_pos': [], 'theta_neg':[],
           'phi_fit':[], 'phi_pos':[], 'phi_neg':[],
           'z_fit':[], 'z_pos':[], 'z_neg':[],
           'multisphere_covar': [],
           'multispherefit_time':[]}


for i in range(len(bd['imagename'])):
    fname = bd['imagename'][i]
    thetaguess = bd['theta_fit'][i]
    zguess = bd['z_fit'][i]*mag
    phiguess = phiguesses[i]
    ##check bisphere for guess first
    #holo, cluster = bisphere(a_p = a_p, n_p = n_p, z_p = zguess, theta = thetaguess, phi = phiguess)
    #plt.imshow(holo)
    #plt.show()
    #print(fname)
    savedir['imagename'].append(fname)
    savedir['a_p'].append(a_p)
    savedir['n_p'].append(n_p)

    
    imagepath = imagebase + fname
    img = cv2.imread(imagepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float)
    img /= np.mean(img)
    start = time()
    a,n,z = fit(img, a_p=a_p*2, n_p=n_p, z_p=zguess, percentpix=1)
    delta = time() - start

    savedir['sphere_a_fit'].append(a)
    savedir['sphere_n_fit'].append(n)
    savedir['sphere_z_fit'].append(z)
    savedir['spherefit_time'].append(delta)

    start = time()
    result = fit_multisphere(imagepath, a_p = a_p, n_p = n_p, theta_guess = thetaguess, z_guess = zguess, phi_guess = phiguess)
    delta = time() - start
    best_fit_values = result.intervals
    theta_unc = [x for x in best_fit_values if x.name=='theta'][0]
    phi_unc = [x for x in best_fit_values if x.name=='phi'][0]
    z_unc = [x for x in best_fit_values if x.name=='z_p'][0]
    
    saveholo = np.clip(np.array(result.hologram)*100., 1, 255)
    cv2.imwrite(imagebase + 'fits/'+ fname, saveholo)

    #dear god i hope no one ever has to look at this code, i'm sorry father for i have sinned
    savedir['multisphere_covar'].append(result.mpfit_details.covar.tolist())
    savedir['multispherefit_time'].append(delta)
    savedir['z_fit'].append(z_unc.guess)
    savedir['z_pos'].append(z_unc.plus)
    savedir['z_neg'].append(z_unc.minus)
    savedir['theta_fit'].append(theta_unc.guess)
    savedir['theta_pos'].append(theta_unc.plus)
    savedir['theta_neg'].append(theta_unc.minus)
    savedir['phi_fit'].append(phi_unc.guess)
    savedir['phi_pos'].append(phi_unc.plus)
    savedir['phi_neg'].append(phi_unc.minus)


with open('data/exp_multisphere_fits04.json', 'w') as f:
    json.dump(savedir, f)
