# Highell_Cls

provide tools to compute high multipole (small-scale) angular power spectra of the thermal and kinetic Sunyaev-Zel'dovich effects. Examples of how to obtain such spectra are given in the tutorial notebook. Note that the first execution of the notebook or the code will download the RF coefficient files (~200Mb each) and will be much slower than futher executions.  

## tSZ angular power spectra

The tSZ Cls are computed with a Random Forest network (see Douspis et al. 2022).

* Highell_Cls.Cl_tsz_fromRF(dictionary) returns ell, Cls between l=2, l=13500 with Cls in muK2 at 143GHz. The dictionary contains cosmological and scaling relation parameters such as: C_SR_params={"bias": 0.8,
                        "H0": 64,
                        "omb": 0.05,
                        "om": 0.31,
                        "sig8": 0.8,
                        "alpha":1.77})

* Highell_Cls.Cl_tsz_fromRF(dictionary) uses two input files corresponding to the coefficients of the Random Forest (RF-TSZ-2021-3__clf_rf_params.pickle) and the exponents of the parameters dependency (tRF-TSZ-2021-4__exponents.pickle) stored in tszcl_coeff/ If the RF-params is not already in your directory (first timetry) it will be downloaded. 

## kSZ angular power spectra

The late-time and patchy kSZ Cls are computed independently with a Random Forest network, given a set of cosmological and reionisation parameters(see Gorce et al. 2022 for more details).

* Highell_Cls.Cl_ksz_fromRF(dictionary) returns the ell-bins the RF was trained on (between l=2, l=13500), the late-time kSZ, and the patchy kSZ Cls in muK2. The dictionary contains cosmological and sreionisation parameters such as: 
param_dict={"omegabh2": 0.0224,
            "omegach2": 0.120,
            "ns": 0.9665,
            "theta": 1.041,
            "logA": 3.044,
            "zre": 7.5,
            "dz": 1.2,
            "logalpha0": 3.7,
            "kappa": 0.10}

## Licence

This project is licensed under the LGPL+3.0 licence.

## Acknowledgements

Please acknowledge the use of the code by citing: Douspis, Salvati, Gorce, Aghanim, A&A, 659, A99, 2022 and Gorce, Douspis, Salvati, A&A 2022, as well as adding the following in the acknowledgements: "This work used the RF-High-ell spectra python code available at szdb.osups.universite-paris-saclay.fr". 
