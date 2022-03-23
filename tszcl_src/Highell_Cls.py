# Copyright (c) 2022 IAS / CNRS / Univ. Paris-Saclay
# CC BY-NC License 
# Author: Marian Douspis <marian.douspis@ias.u-psud.fr>
# DOI: https://doi.org/10.48326/idoc.cosmo.ml-sz-aps


def Cl_tsz_fromRF(C_SR_params={"bias": 0.8,
                        "H0": 64,
                        "omb": 0.05,
                        "om": 0.31,
                        "sig8": 0.8,
                               "alpha":1.77}, coeff_path="../tszcl_coeff/"):
    
    '''Return ell, Cl_TSZ from a set of cosmological and scaling relation parameters

    C_SR_params: Cosmological and Scaling Relation parameters
                 Dictionnary containing the following keys:
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
    output: ell in [2,13500], Cl_tSZ in muK2 at 143GHz

    '''

    import pickle
    import numpy as np
    import scipy as sp
    import requests
    import os.path

    # December 2021
    seed_coeff     =  "RF-TSZ-2021-3"
    seed_exponents =  "RF-TSZ-2021-4"


    file4coeff = coeff_path+seed_coeff+"__clf_rf_params.pickle"
    
    if not os.path.exists(file4coeff):
        url = 'https://www.ias.u-psud.fr/douspis/RF-TSZ-2021-3__clf_rf_params_dld.pickle'
        r = requests.get(url, allow_redirects=True)

        open(file4coeff, 'wb').write(r.content)
    
    
    rf             = pickle.load(
        open(file4coeff, 'rb'))
    expo           = pickle.load(
        open(coeff_path+seed_exponents+"__exponents.pickle", 'rb'))

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
