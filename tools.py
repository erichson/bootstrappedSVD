import numpy as np
from scipy.sparse.linalg import svds

def compute_sketch_(X, l, sampling_method='uniform'):
        m, n = X.shape    
        if(sampling_method == 'uniform'):
            idx = np.random.choice(range(m), l, True)
            Y = X[idx,:] * ( 1 / np.sqrt(l / m))
        elif(sampling_method == 'energy'):
            p = np.linalg.norm(X, 2, axis=1)**2 / np.linalg.norm(X)**2 
            idx = np.random.choice(range(m), l, True, p)
            Y = X[idx,:] / np.array(np.sqrt(l * p[idx])).reshape(l,1)       
        elif(sampling_method == 'gaussian'):
             Omega = np.random.normal(loc = 0.0, scale = 1, size = (l, m))
             Omega *= 1 / np.sqrt(l)
             Y = Omega.dot(X) 
        return Y 

def error_metric_(u, s, vt, shat, vthat, k=1, metric = 'singular_values'):
    if metric == 'singular_values':
        return np.abs(s[(k-1)]-shat[(k-1)])
    elif metric == 'left_singular_vectors':
        z = 1 - s[(k-1)]**2/shat[(k-1)]**2 * (vt[(k-1),:].T.dot(vthat[(k-1),:]))**2 
        z = np.sqrt(z)
        return z
    elif metric == 'right_singular_vectors':
        z = 1 - np.diag((vt.dot(vthat.T)))[k-1]**2
        z = np.sqrt(z)
        return z

def partial_svd_(X, k):
    u, s, vt = svds(X, k)
    if k>1:
        u[ : , :k ] = u[ : , k-1::-1 ]
        s = s[ ::-1 ]
        vt[ :k , : ] = vt[ k-1::-1 , : ] 
    return u, s, vt