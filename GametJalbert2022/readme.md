
# A flexible extended generalized Pareto distribution for tail estimation

Philmon Gamet$^{1,2}$ & Jonathan Jalbert$^2$

$^1$Descartes Underwriting  
$^2$Polytechnique Montréal


This program goes along with the paper of Gamet & Jalbert (2022) available *via* this [link](https://doi.org/10.1002/env.2744). This program defines the extended generalized Pareto distributions described in the paper and provides the code to reproduce the results.

Requirements : 
- Julia 1.6 and newer;
- Jupyter notebook.

<div class="alert alert-block alert-info">
<b>Note :</b> The Extremes.jl and ExtendedExtremes.jl libraries are currently being refactored. That's why the versions used in this notebook are different from those of the master branches. 
</div>


### Reference
Gamet, P. & Jalbert, J. (2022). A flexible extended generalized Pareto distribution for tail estimation. *Environmetrics*, 33(6), e2744.


## Code

The code is available in a Jupiter notebook with a Julia kernel. The principal file is *IDF_interpolation-Example.ipynb*. If GitHub has difficulties to render it for online viewing, you can paste the url in the following Jupyter viewer: https://nbviewer.jupyter.org/

To able the run the code, Julia 1.6 or newer should be installed along with the following Packages:

- CSV
- DataFrames
- Distributions
- ProgressMeter