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

with open('data/exp_multisphere_fits02.json', 'r') as f:
    bd = json.load(f)

n_p = 1.435
a_p = 0.527

image_fnames = ['crop01_square.png', 'crop02.png', 'crop03.png', 'crop04.png']
imagebase = 'data/exp_crops/'

savedir = {'a1_fit': [], 'a2_fit': [], 'n_p': [], 'imagename': [],
           'theta_fit': [], 'phi_fit':[], 'z_fit':[],
           'a1_pos': [], 'a1_neg':[],
           'a2_pos':[], 'a2_neg':[],
           'z_pos': [], 'z_neg':[],
           'theta_pos':[], 'theta_neg':[],
           'phi_pos':[], 'phi_neg':[],
           'covar':[], 'fnorm':[],
           'fittime':[]}


for i in range(len(bd['imagename'])):
    savedir['n_p'].append(n_p)
    fname = bd['imagename'][i]
    thetaguess = bd['multisphere_theta_fit'][i]
    zguess = bd['multisphere_z_fit'][i]
    phiguess = bd['multisphere_phi_fit'][i]
    print(fname)
    savedir['imagename'].append(fname)
    
    imagepath = imagebase + fname

    start = time()
    result = fit_multisphere(imagepath, a_p = a_p, n_p = n_p, theta_guess = thetaguess, z_guess = zguess, phi_guess = phiguess, fit_a=True)
    delta = time() - start
    '''
    best_fit_values = result.parameters


    saveholo = np.clip(np.array(result.hologram)*100., 1, 255)
    cv2.imwrite(imagebase + 'fits/5param/'+ fname, saveholo)
    print(best_fit_values)
    
    savedir['theta'].append(best_fit_values['theta'])
    savedir['phi'].append(best_fit_values['phi'])
    savedir['z_p'].append(best_fit_values['z_p'])
    savedir['a_1'].append(best_fit_values['a_1'])
    savedir['a_2'].append(best_fit_values['a_2'])
    #savedir['multisphere_redchi'].append(result.chisq)
    savedir['fittime'].append(delta)'''

    best_fit_values = result.intervals
    theta_unc = [x for x in best_fit_values if x.name=='theta'][0]
    phi_unc = [x for x in best_fit_values if x.name=='phi'][0]
    z_unc = [x for x in best_fit_values if x.name=='z_p'][0]
    a1_unc =  [x for x in best_fit_values if x.name=='a_1'][0]
    a2_unc =  [x for x in best_fit_values if x.name=='a_2'][0]

    saveholo = np.clip(np.array(result.hologram)*100., 1, 255)
    cv2.imwrite(imagebase + 'fits/5param/'+ fname, saveholo)

    #dear god i hope no one ever has to look at this code, i'm sorry father for i have sinned         
    savedir['covar'].append(result.mpfit_details.covar.tolist())
    savedir['fnorm'].append(result.mpfit_details.fnorm)
    savedir['fittime'].append(delta)
    savedir['z_fit'].append(z_unc.guess)
    savedir['z_pos'].append(z_unc.plus)
    savedir['z_neg'].append(z_unc.minus)
    savedir['theta_fit'].append(theta_unc.guess)
    savedir['theta_pos'].append(theta_unc.plus)
    savedir['theta_neg'].append(theta_unc.minus)
    savedir['phi_fit'].append(phi_unc.guess)
    savedir['phi_pos'].append(phi_unc.plus)
    savedir['phi_neg'].append(phi_unc.minus)
    savedir['a1_fit'].append(a1_unc.guess)
    savedir['a1_pos'].append(a1_unc.plus)
    savedir['a1_neg'].append(a1_unc.minus)
    savedir['a2_fit'].append(a2_unc.guess)
    savedir['a2_pos'].append(a2_unc.plus)
    savedir['a2_neg'].append(a2_unc.minus)


with open('data/exp_multisphere_fits_5param02.json', 'w') as f:
    json.dump(savedir, f)
