import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tools import partial_svd_, compute_sketch_, error_metric_
from scipy.linalg import svd

from sklearn.utils import check_random_state

class sketchedSVD:
    """Sketched Singular Value Decomposition (sketchedSVD). 
    
    Parameters
    ----------
    n_components : int
        Number of components to compute.
    store_sketch : bool, default=True
        If False, sketch is not stored.
    sampling_method : str {'uniform', 'energy', 'gaussian'}
        If uniform :
            Uniform sampling of rows.
        If energy :
            Squared length sampling of rows
        If gaussian :
            Gaussian random projections

    random_state : integer, RandomState instance or None, optional (default ``None``)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    Attributes
    ----------
    right_singular_vectors_ : array, shape (n_components, n)
        
    singular_values_ : array, shape (n_components,)
    """
    
    def __init__(self, n_components=None, store_sketch=True, sampling_method='uniform',
                 n_oversamples=None, random_state=None):
        self.n_components = n_components
        self.store_sketch = store_sketch
        self.sampling_method = sampling_method
        self.n_oversamples = n_oversamples
        self.random_state = random_state
        self.sketch = None
        self.left_singular_vectors_ = None
        self.right_singular_vectors_ = None
        self.singular_values_ = None

        self.random_state = check_random_state(random_state)
         
    def fit(self, X, left_singular_vector=True, ortho=True):
        k = self.n_components
        self.n_samples, self.n_features  = X.shape
        if k >= self.n_samples or k >= self.n_features:
                raise ValueError("n_components must be < min{m,n}")
                
        if self.n_oversamples is None:
            self.n_oversamples = int(np.floor(self.n_samples/2.0)) + self.n_components
        else:
            self.n_oversamples = np.minimum(self.n_samples, self.n_oversamples)
        
        self.sketch = compute_sketch_(X, self.n_oversamples, self.sampling_method, self.random_state)
        self.left_singular_vectors_, self.singular_values_, self.right_singular_vectors_ = partial_svd_(self.sketch, self.n_components)
    
        if left_singular_vector==True:
                self.left_singular_vectors_ = X.dot(self.right_singular_vectors_.T  *  self.singular_values_**-1) 
                if ortho==True:
                    self.left_singular_vectors_ , self.singular_values_ , Vt = svd( self.left_singular_vectors_*self.singular_values_,
                                            full_matrices=False, overwrite_a=True, check_finite=True)
                    self.right_singular_vectors_ = Vt.dot(self.right_singular_vectors_)  
                    
        return self   
    
    def reconstruct(self):
        return (self.left_singular_vectors_*self.singular_values_).dot(self.right_singular_vectors_)
    
    def error_estimation(self, top_n_components=1, bootstrap_replicates=100, metric = 'singular_values'):
        self.error_estimates = []
        
        self.top_n_components = np.minimum(top_n_components, self.n_components)
        print(self.top_n_components)
        
        if self.n_oversamples <= 100:
            raise ValueError("The sketch is too small to compute meaningful error estimates.")
        
        #for i in range(bootstrap_replicates):
        for i in tqdm(range(bootstrap_replicates)):
            t = True
            while(t):
                row_index_set = self.random_state.choice(range(self.n_oversamples), self.n_oversamples, replace=True, p=None)    
                Z = self.sketch[row_index_set, :]
                u_boot, s_boot, vt_boot = partial_svd_(Z, self.top_n_components)
                
                if metric == 'left_singular_vectors':
                    s_boot = np.asarray(np.linalg.norm(self.sketch.dot(vt_boot.T), 2, axis=0))
                
                w = [error_metric_(self.left_singular_vectors_, self.singular_values_, self.right_singular_vectors_, 
                               s_boot, vt_boot, k=k+1, metric = metric) for k in range(self.top_n_components)]
                if np.isnan(np.sum(w)) == False:
                    t = False
            self.error_estimates.append(w)
        self.error_estimates = np.asarray(self.error_estimates)    
        return self

    def error_estimation_plot(self, alpha=0.1, interval=None):
        if interval is None:
            lower = 500; upper= self.n_samples
        else:
            lower = interval[0]; upper = interval[1]
        
        errors = np.percentile(self.error_estimates, (1-alpha)*100, axis=0).reshape(1, self.top_n_components)
        sampling_range = np.arange(lower, upper, 1)
        scaling = np.sqrt(self.n_oversamples / sampling_range).reshape(len(sampling_range),1)
        error_extrapol =  errors * np.repeat(scaling, scaling.shape[1], axis=1)
        
        for i in range(self.top_n_components):
            plt.figure(figsize=(8,4))
            plt.title('Component %i' %(i+1))
            plt.axvline(self.n_oversamples, label='sketch size')
            plt.plot(sampling_range, error_extrapol[:,i], lw=4, color='#de2d26', label='error extrapolation')
            plt.ylabel('error', fontsize=14)
            plt.xlabel('sketch size', fontsize=14)
            plt.legend(loc="best", fontsize=12)
            plt.tick_params(axis='y', labelsize=12)
            plt.tick_params(axis='x', labelsize=12)     
            plt.tight_layout()
            plt.show()