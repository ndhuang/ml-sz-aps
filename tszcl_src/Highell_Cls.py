# Copyright (c) 2022 IAS / CNRS / Univ. Paris-Saclay
# LGPL License - see attached LICENSE file
# Authors: Marian Douspis <marian.douspis@ias.u-psud.fr>
#          Adélie Gorce <adelie.gorce@gmail.com>

import pickle
import sklearn
import numpy as np
import warnings
import os
import scipy as sp
import requests

def Cl_tsz_fromRF(C_SR_params={"bias": 0.8,
                               "H0": 64,
                               "omb": 0.05,
                               "om": 0.31,
                               "sig8": 0.8,
                               "alpha":1.77},
                  coeff_path="../tszcl_coeff/"):
    
    '''Return ell, Cl_TSZ from a set of cosmological and scaling relation parameters

        Parameters
        ----------
        C_SR_params: Dictionary
                Cosmological and Scaling Relation parameters gathered in a 
                dictionary containing the following keys:
                 H0   : Hubble constant in [60,80]
                 omb  : Omega_baryon in [0.02,0.07]
                 om   : Ommega_matter in [0.2, 0.4]
                 sig8 : sigma_8  in [0.7, 0.9]
                 bias : hydrostatic mass bias in [0.6,0.99]
                 alpha: mass exponent in scaling relation in [1.75, 1.82]
                 c_sr_example = {
                                 "bias": 0.8,
                                 "H0": 64,
                                 "omb": 0.05,
                                 "om": 0.31,
                                 "sig8": 0.8,
                                 "alpha":1.77
                                  }
      coeff_path: string
               Path to folder containing the RF coefficient file.
               
      Returns
      ----------
      ells: list of floats
          List of angular multipoles the power spectrum has been 
          calculated at (ell in [2,13500])
      Cl_tSZ: list of floats
          List of angular power spectrum values at 143GHz in muK2

    '''

    # tests
    assert isinstance(C_SR_params, dict), \
        "Must feed cosmological parameters as dictionary - see docstring."
    assert isinstance(coeff_path, str), \
        "Must feed path to RF coefficients as a string"

    file4coeff = "{}RF-TSZ-2021-3__clf_rf_params.pickle".format(coeff_path)
    if not os.path.exists(file4coeff):
        url = 'https://www.ias.u-psud.fr/douspis/RF-TSZ-2021-3__clf_rf_params.pickle'
        r = requests.get(url, allow_redirects=True)
        open(file4coeff, 'wb').write(r.content)
        warnings.warn('Retrieving RF files from {}'.format(url))
    rf             = pickle.load(open(file4coeff, 'rb'))

    file4expo = "{}RF-TSZ-2021-4__exponents.pickle".format(coeff_path)
    assert os.path.exists(file4expo), \
        "File with exponents missing"
    expo           = pickle.load(open(file4expo, 'rb'))

    paramsref = [0.80E+00,
                 -0.2088383E+00,
                 0.1778349E+01,
                 0.1304860E-09,
                 0.9612,
                 0.69E+00,
                 0.5E-01,
                 0.26E+00,
                 0.69E+00,
                 0.80E+00]
    
    params    = np.copy(paramsref)
    params[0] = C_SR_params["bias"]
    params[5] = C_SR_params["H0"]/100.
    params[6] = C_SR_params["omb"]
    params[7] = C_SR_params["om"]-C_SR_params["omb"]
    params[8] = 1.-C_SR_params["om"]
    params[9] = C_SR_params["sig8"]
    params[2] = C_SR_params["alpha"]
    
    output = np.array((rf.predict(params.reshape(1, -1)))[0])
    
    ll     = np.array([10, 400, 1200, 2000, 2800,
                       3600, 5000, 7000, 9000, 10000])
    
    new_xx = np.linspace(2, 13500, 13499)

    output2 = np.exp(
        sp.interpolate.interp1d(np.log(ll), np.log(output),
                                kind='quadratic',
                                fill_value='extrapolate')(np.log(new_xx)))

    output2 = np.exp(output2)*(params[0]**expo[0]
                               * params[1]**expo[1]
                               * params[2]**expo[2]
                               * params[3]**expo[4]
                               * params[4]**expo[5]
                               * params[5]**expo[6]
                               * params[6]**expo[7]
                               * (params[7]+params[6])**expo[8]
                               * params[8]**expo[9]
                               * params[9]**expo[10])
                      
    return new_xx, output2

def Cl_ksz_fromRF(param_dict={"omegabh2": 0.0224,
                              "omegach2": 0.120,
                              "ns": 0.9665,
                              "theta": 1.041,
                              "logA": 3.044,
                              "zre": 7.5,
                              "dz": 1.2,
                              "logalpha0": 3.7,
                              "kappa": 0.10},
                  coeff_path="./kszcl_coeff/"):
    
    '''Return ell, Cl_KSZ from a set of cosmological and scaling relation parameters

      Parameters
      ----------
      param_dict: Dictionary
                  Cosmological and reionisation parameters gathered in a 
                  dictionary containing the following keys:
                     omegabh2   : Density of baryons at z=0 scaled by h^2
                     omegach2  :Density of cold DM at z=0 scaled by h^2
                     ns   : Scalar spectral index
                     theta : Ratio of the sound horizon to the angular diameter distance 
                             at decoupling, scaled by 100
                     logA : ln(10^10 As) for As the amplitude of initial perturbations
                     zre : Midpoint of reionisation
                     dz : zre-zend, where zend is the endpoint of reionisation
                     logalpha0 : decimal logarithm of the large-scale amplitude of the 
                              electron power spectrum
                     kappa : drop-off frequency of the power spectrum of free electrons 
                             density fluctuations
      coeff_path: string
               Path to folder containing the RF coefficient file.
               
      Returns
      ----------
      ells: 1D array of floats
          List of angular multipoles the power spectrum has been 
          calculated at (ell in [2,13500]).
      Cl_kSZ late-time: array of floats
          Late-time kSZ angular power spectrum values in muK2.
      Cl_kSZ patchy: array of floats
          Patchy kSZ angular power spectrum values in muK2.

    '''
    
    # tests
    assert isinstance(param_dict, dict), \
        "Must feed cosmological parameters as dictionary - see docstring."
    assert isinstance(coeff_path, str), \
        "Must feed path to RF coefficients as a string"
    assert os.path.exists("{}RF-KSZ_ells.txt".format(coeff_path)), \
        "File necessary to read RF missing"



    file4coeff = "{}RF-KSZ_patchy_cl_rf.pickle".format(coeff_path)
    if not os.path.exists(file4coeff):
        url = 'https://www.ias.u-psud.fr/douspis/RF-KSZ_patchy_cl_rf.pickle'
        r = requests.get(url, allow_redirects=True)
        open(file4coeff, 'wb').write(r.content)
        warnings.warn('Retrieving RF files patchy from {}'.format(url))

    file4coeff = "{}RF-KSZ_late_cl_rf.pickle".format(coeff_path)
    if not os.path.exists(file4coeff):
        url = 'https://www.ias.u-psud.fr/douspis/RF-KSZ_late_cl_rf.pickle'
        r = requests.get(url, allow_redirects=True)
        open(file4coeff, 'wb').write(r.content)
        warnings.warn('Retrieving RF files late from {}'.format(url))


    
    # convert parameter dictionary to array
    paramnames = ['omegabh2', 'omegach2', 'ns', 'theta', 'logA',
                  'zre', 'dz', 'logalpha0', 'kappa']
    paramlims = [(0.0200,0.0300), (0.100,0.150), (0.957,0.990), (1.03,1.05), (2.8,3.3),
                 (5. ,10.), (0.1,5.), (2.,4.5), (0.04,0.20)]
    
    ndeg = len(paramnames)
    params = np.zeros(ndeg)
    for u, param in enumerate(paramnames):
        assert param in list(param_dict.keys()), \
            "Missing parameter in dictionary: {}".format(param)
        params[u] = float(param_dict[param])
        if not (paramlims[u][0] <= params[u] <= paramlims[u][1]):
            warnings.warn('Value of parameter {} outside of training range.'.format(param))
            
    # Read files
    # Angular multipoles the RF has been trained on
    ll = np.loadtxt("{}RF-KSZ_ells.txt".format(coeff_path))
    # Patchy KSZ coefficients
    rfp = pickle.load(open("{}RF-KSZ_patchy_cl_rf.pickle".format(coeff_path), 'rb')) 
    expop, params_ref = np.loadtxt("{}RF-KSZ_patchy_exponents.txt".format(coeff_path), unpack=True)
    # Late-time KSZ coefficients
    rfh = pickle.load(open("{}RF-KSZ_late_cl_rf.pickle".format(coeff_path), 'rb')) 
    expoh, params_ref = np.loadtxt("{}RF-KSZ_late_exponents.txt".format(coeff_path), unpack=True)

    # patchy
    pksz = rfp.predict(params.reshape(1, ndeg)).flatten()
    pksz = pksz*expop[0]*np.product(np.abs(params/params_ref[1:])**expop[1:])
    pksz[pksz<0] = 0.
    
    # late-time
    hksz = rfh.predict(params.reshape(1, ndeg)).flatten()
    hksz = hksz*expoh[0]*np.product(np.abs(params/params_ref[1:])**expoh[1:])
    hksz[hksz<0] = 0.

    return ll, hksz, pksz

