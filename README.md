# Error Estimation for Sketched SVD via the Bootstrap

This repository provides research code for a data-driven bootstrap method that numerically estimates the actual error of sketched singular vectors and values.
This allows the user to inspect the quality of a rough initial sketched Singular Value Decomposition (SVD), and then adaptively predict how much extra work is needed to reach a given error tolerance. 
To demonstrate the idea, we generate a synthetic matrix which is characterized by a low effective rank.
```
from sketchedSVD import *
from genTestMatrix import *         

X = matrixPolyDecay(m=100000, n=3000, beta=1)    
``` 
Here, `X` is a tall and skinny matrix which features a polynomially decaying spectrum. (Generating the data will take a little while...)

In the following we are interested in computing an approximation for the top right singular singular vector and value (`n_components=1`). The quality of the approximation depends on the sketch size. However, in practice, it is often difficult to determine a `good' sketch size in absence of any prior knowledge. 
By `good' we mean a sketch size that provides a good trade-off between accuracy and computational efficiency. 
To gain some insights, we compute a rough initial sketched SVD. Therefore, we construct a tiny sketch by uniformaly sampling 1% of the rows `n_oversamples=1000`.
```
svd = sketchedSVD(n_components=1, n_oversamples=1000, sampling_method='uniform', random_state=1)
svd.fit(X, left_singular_vector=True)
``` 
Note that our method is not restricted to uniform sampling. You can also use squared length sampling (`sampling_method='energy'`) or Gaussian projections (`sampling_method='gaussian'`) or any other random sampling strategy.
But, here we know that the information are uniformly distributed across the data matrix and thus uniform sampling is a perfectly fine option for sampling. Btw., it is also the most computational efficient sampling strategy. Fun fact: It is very efficient to read a small subset of rows from an external storage device into the fast memory if your data matrix is stored in some hierarchical data format. (There is no need to touch the full data matrix if you set `left_singular_vector=False`.)

Now, let's check how good this (small) initial sketch is by running the bootstrap method (the computations will be super quick, i.e., less than 3 seconds on my notebook).
```
svd.error_estimation(top_n_components=1, bootstrap_replicates=40, metric = 'right_singular_vectors')
```
The bootstrap method is embarrassingly parallel and it is easy to parallelize the computations, however, depending on the computational environment, the additional overhead costs introduced by the parallel computing routines can outweigh the computational benefits for the particular problem at hand.

Next, we use the results to extrapolate (forecast) the error as a function of the sketch size.
```
svd.error_estimation_plot(alpha=0.1, interval=[500,10000])

```
You will generate the following plot which tells you that you can roughly reduce the error by a factor of 2 if you sample 4% to 6% of the rows. 
Further, we see that increasing the sketch even further has a diminishing return in terms of the approximation accuracy.  

<img src="https://github.com/erichson/bootstrappedSVD/blob/master/plots/eestimate.png" width="800">


You don't believe that these error estimates are accurate? Well, you can verify that the extrapolated errors are actually pretty accurate by running the following simulation
```
exec(open('simulation.py').read())
```
<img src="https://github.com/erichson/bootstrappedSVD/blob/master/plots/simulation.png" width="800">

Our paper provides many more details, experimental results and theoretical guarantees for the proposed bootstrap method.


### Reference
[Lopes et al., Error Estimation for Sketched SVD via the Bootstrap, ICML (2020)](https://arxiv.org/pdf/2003.04937.pdf)


