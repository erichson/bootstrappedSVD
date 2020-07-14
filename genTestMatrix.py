import numpy as np

def matrixPolyDecay(m, n, beta, seed=1):
    np.random.seed(seed)
    
    U, _ = np.linalg.qr(np.random.standard_normal((m,n)))
    V, _ = np.linalg.qr(np.random.standard_normal((n,n)))
        
    sn = np.minimum(m,n)
    s = np.float64(np.arange(1,sn+1))
    s = s ** -beta

    X = (U*s).dot(V.T)
        
    if m < n:
        return X.T
    else:
        return X
    
def matrixExpDecay(m, n, beta, seed=1):
    np.random.seed(seed)
    
    U, _ = np.linalg.qr(np.random.standard_normal((m,n)))
    V, _ = np.linalg.qr(np.random.standard_normal((n,n)))
        
    sn = np.minimum(m,n)
    s = np.float64(np.arange(1,sn+1))
    s = 10 ** (-beta * s)
    s[0] = 1
        
    X = (U*s).dot(V.T)
   
    if m < n:
        return X.T
    else:
        return X
    
    
    
    