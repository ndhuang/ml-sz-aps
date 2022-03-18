# Highell_Cls

Provide tools to compute  High ell angular power spectra of SZ related spectra. 

## tSZ Cls spectra

The tSZ Cls are computed thanks to a Random Forest trained network (see [Douspis et al. 2022](https://www.aanda.org/articles/aa/full_html/2022/03/aa42004-21/aa42004-21.html).

* Highell_Cls.Cl_tsz_fromRF(dictionary) returns ell, Cls between l=2, l=13500 with Cls in muK2 at 143GHz. The dictionary contains cosmological and scaling relation parameters such as: C_SR_params={"bias": 0.8,
                        "H0": 64,
                        "omb": 0.05,
                        "om": 0.31,
                        "sig8": 0.8,
                        "alpha":1.77})

* Highell_Cls.Cl_tsz_fromRF(dictionary) uses two input files corresponding to the coefficients of the Random Forest (RF-TSZ-2021-3__clf_rf_params.pickle) and the exponents of the parameters dependency (tRF-TSZ-2021-4__exponents.pickle) stored in tszcl_coeff/. If the RF-params is not already in your directory (first time try) it will be downloaded. 

## Python notebook

The Highell_Cls.ypnb file allows you to see how to use the code with a few examples.

## Licence

This project is licensed under the Common Creative CC BY-NC.

## Acknowledgements

Please acknowledge the use of the code by citing: [Douspis, Salvati, Gorce, Aghanim, A&A, 659, A99, 2022](https://www.aanda.org/articles/aa/full_html/2022/03/aa42004-21/aa42004-21.html) and   
adding the following in the acknowledgements: "This work used the RF-High-ell spectra python code available at szdb.osups.universite-paris-saclay.fr"

## Next

Coming soon, the kSZ spectrum computation based on [Gorce et al. 2020](https://arxiv.org/abs/2004.06616) and [Gorce et al. 2022](https://arxiv.org/abs/2202.08698)
