# GRB_LIV_Hierarchical
This repository includes the spectral-lag data of 32 GRBs used in the literature. 

The data sets of spectral lags of 32 GRBs can be found in the 'Data/lag_err_fermi_32grbs/' directory. In each 'GRBName.txt' file, we list the medium of each energy band adopted to calculate the time lags (with using a lowest energy band as reference), the 1sigma error of each energy band, the time lag, and the 1sigma error of time lag. 

In 'GRBPARAM.csv', we privide the entire energy band adopted to extract the Fermi-GBM light curves of each GRB, and the redshift of each burst. 

The posterior samples of log_10(E_{\rm QG,n}) for individual GRBs can be found in the 'Data/posteriors_SBPL_L22' (SBPL approach) and 'Data/poteriors_cubicspline_DU25' (Cubic-Spline approach). 

Liu, Z.-K. performed the data analyses in extracting the spectral lags of those GRBs, which have been used to constrain quantum gravity energy scales from each GRB, see Liu et al. 2022, APJ, 935:79(8pp) (https://doi.org/10.3847/1538-4357/ac81b9). 


