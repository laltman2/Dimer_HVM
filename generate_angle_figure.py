import holopy as hp
from holopy.scattering import Scatterer, Sphere, Spheres, calc_holo
from holopy.scattering.scatterer import Indicators
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import json
from pylorenzmie.theory import coordinates, LMHologram
from pylorenzmie.analysis.Feature import Feature
from holopy.scattering.theory import DDA
from cluster_holo import *

#Instrument parameters:
wv = 0.447
mag = 0.120
n_m = 1.34 #assume water
px = 200

config = {'n_m': n_m, 'wavelength': wv, 'magnification': mag}

n_p = 1.435 #Silica
a_p = 0.527 #um

num_to_sample = 1000

samples = {'z_p': [], 'n_p': n_p, 'a_p': a_p, 'phi': [], 'theta': [], 'a_p^*': [], 'n_p^*': [], 'z_p^*':[]}
for i in range(num_to_sample):
    z_p = np.random.uniform(50, 95)
    samples['z_p'].append(z_p)
    print('dimer number {}'.format(i))
    theta = np.random.uniform(low=0, high=np.pi/2)
    print('theta = {}'.format(str(theta).zfill(3)))
    phi = np.random.uniform(low=0, high=np.pi)
    data, _ = bisphere(a_p, n_p, z_p, theta=theta, phi=phi)
    a, n, z = fit(data, a_p, n_p, z_p)

    samples['theta'].append(theta)
    samples['phi'].append(phi)
    samples['a_p^*'].append(a)
    samples['n_p^*'].append(n)
    samples['z_p^*'].append(z)


with open('./data/angle_dimer_fits_121120.json', 'w') as f:
    json.dump(samples, f)
