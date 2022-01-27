


# Interpolation of precipitation extremes on a large domain toward IDF curve construction at unmonitored locations

Jonathan Jalbert$^1$, Christian Genest$^2$ and Luc Perreault$^3$

$^1$Polytechnique Montréal  
$^2$McGill University  
$^3$Institut de recherche d'Hydro-Québec  


## Description

This program goes along with the paper of Jalbert *et al.* (2022) which describes how extreme precipitation of several durations can be interpolated to compute IDF curves on a large, sparse domain. In this code, sparse precipitation extremes for a given duration are interpolated on a regular lattice. The lattice is the one where the spatial covariate lies. The cross-validation program is performed in another file.

## Code

The code is available in a Jupiter notebook with a Julia kernel. The principal file is *IDF_interpolation-Example.ipynb*. If GitHub has difficulties to render it for online viewing, you can paste the url in the following Jupyter viewer: https://nbviewer.jupyter.org/

To able the run the code, Julia 1.6 or newer should be installed along with the following Packages:

- CSV
- Distributions
- Gadfly
- Extremes
- Mamba
- ProgressMeter
- Plots
- StatsBase

The unregisterred package *GMRF.jl$ is also required. To install it, run in the Julia package manager the following command:

    add https://github.com/jojal5/GMRF.jl


## Reference

Jalbert J., Genest, C. and Perreault L, (2022). Interpolation of precipitation extremes on a large domain toward IDF curve construction at unmonitored locations. *Journal of Agricultural, Biological, and Environmental Statistics*, To appear.