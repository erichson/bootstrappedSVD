import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import timeit
import argparse
import os 

from tools import *


def boot(Y, u, s, vt, k, metric):
    t = True
    while(t):
        l = Y.shape[0]
        idxBoot = np.random.choice(range(l), l, replace=True, p=None)    
        Z = Y[idxBoot, :]
        uboot, sboot, vtboot = partial_svd_(Z, k=k)
    
        if metric == 'left_singular_vectors':
            sboot = np.asarray(np.linalg.norm(Y.dot(vtboot.T), 2, axis=0))
        
        w = error_metric_(u, s, vt, sboot, vtboot, k=k, metric = metric)
        
        if np.isnan(w) == False:
            return w
        

    

def simulation(A, sample, runs, boots, alpha=0.1, sampling_method = 'gaussian', metric = 'sv', k=1):
    print('Start Bootstrap with sample: ', sample)
    m, n = A.shape
    u, s, vt = partial_svd_(A, k)
    
    error = []
    error_boot = []
    print(sample)

    for i in tqdm(range(runs)):
        t = True
        while(t):
            Y = compute_sketch_(A, sample, sampling_method = sampling_method)
            uhat, shat, vthat = partial_svd_(Y,  k=k)
                
            if metric == 'left_singular_vectors':                
                A_shat = np.asarray(np.linalg.norm(A.dot(vthat.T), 2, axis=0))
                w = error_metric_(u, s, vt, A_shat, vthat, k=k, metric = metric)
            else:
                w = error_metric_(u, s, vt, shat, vthat, k=k, metric = metric)
                    
            if np.isnan(w) == False:
                break
   
        error.append(w)
        
        out = [boot(Y, uhat, shat, vthat, k, metric) for i in range(boots)]       
        error_boot.append(out)
    
    return error, error_boot




def plot_results(indx, alpha):

    clevel = 1-alpha
    
    plt.figure(figsize=(8,4))
    plt.plot(samples[indx::], qunatile_e[indx::], 'k--', lw=4, label='true %1.2f-quantile' %clevel)    
    plt.plot(samples[indx::], quantiles_star_mean[indx::], lw=4,
             color='#3182bd', label='avg. bootstrap quantile')

    samples_inter = np.arange(samples[indx], samples[-1], 1)
    scaling = np.sqrt(np.asarray(samples)[indx] / samples_inter)
    error_extrapol = scaling * quantiles_star_mean[indx]
    sd_extrapol = scaling * quantiles_star_sd[indx]
    upperlimits = error_extrapol + 1*sd_extrapol
    lowerlimits = np.maximum(0, error_extrapol - 1*sd_extrapol)
    
    plt.plot(samples_inter, error_extrapol, lw=4,
             color='#de2d26', label=r'extrapolation $\pm 1$')

    plt.fill_between(samples_inter, lowerlimits, upperlimits, 
                     facecolor='b', color='#de2d26', alpha=0.1)

    plt.plot(samples_inter, lowerlimits, lw=2, color='#de2d26', alpha=0.3)
    plt.plot(samples_inter, upperlimits, lw=2, color='#de2d26', alpha=0.3)

    plt.ylabel('error', fontsize=14)
    plt.xlabel('sketch size', fontsize=14)
    plt.legend(loc="best", fontsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=12)     
     
    plt.tight_layout()


if __name__ == "__main__":
    print('Start.........')
    t0 = timeit.default_timer()

    np.random.seed(1234)  
    
    parser = argparse.ArgumentParser(description='Bootstrap Simulation')
    #
    parser.add_argument('--name', type=str, default='test')
    #
    parser.add_argument('--samples', type=int, nargs='+', default=[500, 900, 1500, 2000, 3000])
    #
    parser.add_argument('--runs', type=int, default=200)
    #
    parser.add_argument('--boots', type=int, default=30)
    #
    parser.add_argument('--alpha', type=float, default=0.1)
    #
    parser.add_argument('--sampling_method', type=str, default='uniform') 
    #
    parser.add_argument('--metric', type=str, default='right_singular_vectors') 
    #
    parser.add_argument('--k', type=int, default=1) 
    #    
    args = parser.parse_args()    
    
    #=========================================================================
    # Number of oversamples
    #=========================================================================
    runs = args.runs
    boots = args.boots
    samples = args.samples
    alpha = args.alpha
    sampling_method = args.sampling_method
    metric = args.metric
    
    #=========================================================================
    print('Setting')
    print('***********************')
    print('Matrix dimension', X.shape)
    print('Runs', runs)
    print('Boots', boots)
    print('samples', samples)

    if not os.path.isdir('results'):
        os.mkdir('results')

    #=========================================================================
    qunatile_e = []
    qunatile_e_mean = []

    quantiles_star_mean = []
    quantiles_star_sd = []
    quantiles_star_lower = []    
    quantiles_star_upper = []    
    fail_prob = []
    traps = []
    
    for sample in samples:
        e, e_star = simulation(X, sample, runs, boots, alpha = alpha, sampling_method = sampling_method, metric = metric, k=args.k)
        
        qunatile_e.append((np.percentile(e, (1-alpha)*100)))
        qunatile_e_mean.append(np.mean(e))

        quantiles_star = [(np.percentile(temp, (1-alpha)*100)) for temp in e_star ]
        quantiles_star_mean.append(np.mean(quantiles_star))
        quantiles_star_sd.append(np.std(quantiles_star))   
        quantiles_star_lower.append(np.percentile(quantiles_star, (alpha)*100))   
        quantiles_star_upper.append(np.percentile(quantiles_star, (1-alpha)*100))
        
        store = [qunatile_e, quantiles_star_mean, quantiles_star_sd, quantiles_star_lower, quantiles_star_upper, samples]
        np.save('results/bootstrap_' + args.name +  '.npy', store)

    store = [qunatile_e, quantiles_star_mean, quantiles_star_sd, quantiles_star_lower, quantiles_star_upper, samples]
    np.save('results/bootstrap_' + args.name +  '.npy', store)

    plot_results(0, args.alpha)
    print('Total time:', timeit.default_timer()  - t0 )
