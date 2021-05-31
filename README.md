# Bayesian simulation

This package and notebooks are for testing bayesian approaches to consumer choice tests.

## Notebooks
### [Simple AB choice test](notebooks/bayesian_simulation.ipynb)
In this notebook, we are investigating whether a bayesian optional stopping approach could 
help generate costs savings. The idea is simple:
* each consumer subject chooses between A and B
* we calculate the beta posterior at steps of N subjects
* if the posterior mass strongly (specifically X% HPDI) favors
one of the alternatives strongly enough, break the test early
* assume some cost per subject and a large cost of erroneous inference 
(i.e. deciding that B is better even though A actually is) 

## Future ideas
* More efficient simulation framework
* More complicated test setups
* Interactive cost plots
* Incorporate Kruschke's 
[precision goal](http://doingbayesiandataanalysis.blogspot.com/2013/11/optional-stopping-in-data-collection-p.html?m=1) 
for comparison
* Generating some guidance when it comes to certainty levels and priors
* Thinking about the optimization for different use cases