
import numpy as np
#import spectral_embedding as se
import matplotlib as mpl
import matplotlib.pyplot as plt

from itertools import product
from matplotlib import ticker, cm
from scipy.optimize import minimize

import scipy.stats
import random

from tqdm import tqdm
from sampling import *
from transforms import *

def wsbm_chernoff_information(PI, R, k = 0, l = 1):
    B = np.mean(R, axis = 2)
    C = np.var(R, axis = 2)
    K = len(PI)
    CI = np.zeros((K, K))
    for k in range(K):
        for l in range(K):
            if k != l:
                res = minimize(wsbm_scaled_rd, 0.0, [PI, B, C, k, l], method='TNC', tol=1e-10)
                CI[k,l] = -res.fun
    return CI

#### CI FOR ARBITRARY MODELS ####

def renyi_divergence(t, P, Q):
    t = t[0]
    #print (t, ((P ** t) * (Q ** (1 - t))))
    return -np.log(max(1e-5, ((P ** t) * (Q ** (1 - t))).sum()))

def scaled_rd(x, params):
    t = logit(x)
    P, Q = params
    return -renyi_divergence(t, P, Q)

def discretized_chernoff_information(X, Y, n_buckets = 100):
    P, Q = np.zeros(n_buckets), np.zeros(n_buckets)

    mi, ma = min(X.min(), Y.min()), max(X.max(), Y.max())
    delt = ma - mi
    for x in X: P[min(n_buckets - 1, int(((x - mi)/delt) * n_buckets))] += 1
    for y in Y: Q[min(n_buckets - 1, int(((y - mi)/delt) * n_buckets))] += 1
    P /= len(X); Q /= len(Y)

    #return -R_div_arb(0.5, PI, P, Q)
    res = minimize(scaled_rd, 0.0, [P,Q], method='TNC', tol=1e-10)
    CI = -res.fun
    return CI

def d2_discretized_chernoff_information(X, Y, n_buckets = 5):
    P, Q = np.zeros((n_buckets,n_buckets)), np.zeros((n_buckets,n_buckets))

    mi_x, ma_x = min(X[:,0].min(), Y[:,0].min()), max(X[:,0].max(), Y[:,0].max())
    mi_y, ma_y = min(X[:,1].min(), Y[:,1].min()), max(X[:,1].max(), Y[:,1].max())
    delt_x,delt_y = ma_x - mi_x, ma_y - mi_y
    for x in X:
        P[min(n_buckets - 1, int(((x[0] - mi_x)/delt_x) * n_buckets))][min(n_buckets - 1, int(((x[1] - mi_y)/delt_y) * n_buckets))] += 1
    for y in Y:
        Q[min(n_buckets - 1, int(((y[0] - mi_x)/delt_x) * n_buckets))][min(n_buckets - 1, int(((y[1] - mi_y)/delt_y) * n_buckets))] += 1
    P /= len(X); Q /= len(Y)

    #return -R_div_arb(0.5, PI, P, Q)
    #print ([P, Q])
    res = minimize(scaled_rd, (0.0,), [P.reshape(-1),Q.reshape(-1)], method='TNC', tol=1e-10, bounds = ((0, None),))
    CI = -res.fun
    return CI

#### WEIGHTED SBM ####

def wsbm_renyi_divergence(t, PI, B, C, k, l):
    K = len(PI)
    try: center = B @ PI @ np.linalg.inv(S_matrix(C, k, l, t)) @ B
    except np.linalg.LinAlgError: return 0.0
    vect = standard_basis(K, k) - standard_basis(K, l)
    return t * (1 - t)/2 * vect.T @ center @ vect

def wsbm_scaled_rd(x, params):
    t = logit(x)
    PI, B, C, k, l = params
    return -wsbm_renyi_divergence(t, PI, B, C, k, l)

def wsbm_chernoff_information(PI, R, k = 0, l = 1):
    B = np.mean(R, axis = 2)
    C = np.var(R, axis = 2)
    K = len(PI)
    CI = np.zeros((K, K))
    for k in range(K):
        for l in range(K):
            if k != l:
                res = minimize(wsbm_scaled_rd, 0.0, [PI, B, C, k, l], method='TNC', tol=1e-10)
                CI[k,l] = -res.fun
    return CI

def wsbm_chernoff_information_(PI, B, C, k = 0, l = 1):
    K = len(PI)
    CI = np.zeros((K, K))
    res = minimize(wsbm_scaled_rd, 0.0, [PI, B, C, k, l], method='TNC', tol=1e-10)#, bounds=((0,None),))
    return -res.fun


def wsbm_matrix_CI(PI, DISTS, PARAMS, transform = lambda x : x, rho = 1):
    # PI:     PMF on communities
    # DISTS:  symmetric KxK matrix of distributions for community i -> community j
    # PARAMS: symmetric KxK matrix of parameters for community i -> community j

    N = 100_000
    K = len(PI)

    R = matrix_truncated_sample(DISTS, PARAMS, rho, N)

    R = transform(R)

    R = np.array(R)
    PI = np.diag(PI)
    return wsbm_chernoff_information(PI, R)

def wsbm_scalar_CI(PI, R):
    R = np.array(R)
    PI = np.diag(PI)
    return wsbm_chernoff_information(PI, R)[0,1]




def wsbm_CI_(pi, samples_, transform = lambda x : x, label = None):
    # PI:     PMF on communities
    # DISTS:  symmetric KxK matrix of distributions for community i -> community j
    # PARAMS: symmetric KxK matrix of parameters for community i -> community j

    PI = np.diag([pi, 1-pi])

    B = np.zeros((2,2))
    C = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            samples = transform(samples_[i][j])
            B[i][j] = np.mean(samples)
            C[i][j] = np.var(samples)
    #print (label, B)
    CI = wsbm_chernoff_information_(PI, B, C, 0, 1)
    return CI

def wsbm_CI(pi, DISTS, PARAMS, transform = lambda x : x, rho = 1):
    # PI:     PMF on communities
    # DISTS:  symmetric KxK matrix of distributions for community i -> community j
    # PARAMS: symmetric KxK matrix of parameters for community i -> community j

    N = 10_000_000

    samples = [[0,0],[0,0]]
    for i in range(2):
        for j in range(2):
            samples[i][j] = truncated_sample(DISTS[i][j], PARAMS[i][j], rho, N)
    
    return wsbm_CI_(pi, samples, transform)

def plot_transforms_new(model, transforms, gran=50):
    pi, DISTS, PARAMS,dist_label = model.pi, model.DISTS, model.PARAMS, model.dist_label
    N = 100_000
    res = np.zeros((gran + 1, len(transforms)))

    for rho in tqdm(range(0, gran + 1)):
        samples = [[0,0],[0,0]]
        for i in range(2):
            for j in range(2):
                #if i == 0 and j == 0:
                #    #print (PARAMS[i][j], rho/gran)
                samples[i][j] = truncated_sample(DISTS[i][j], PARAMS[i][j], rho/(gran), N)
        samples=np.array(samples)
        #print(samples.shape)
        CIs = []
        for transform in transforms:
            CIs.append(wsbm_CI_(pi, samples, transform[0], transform[1]))
        res[rho] = np.array(CIs)
    
    for i in range(len(transforms)):
        plt.plot([x / gran for x in range(0,gran + 1)], res[:,i], label=transforms[i][1],alpha=0.5)
    plt.xlabel("$\\rho$")
    plt.ylabel("chernoff information")
    plt.legend()
    plt.title(f"CI against $\\rho$ for $U[0,1]$ vs. {dist_label}")
    plt.savefig("gammalines.svg")

def plot_contour_gamma(model, transforms, rho, gran=30, symm=False):
    pi, DISTS, PARAMS,dist_label = model.pi, model.DISTS, model.PARAMS, model.dist_label
    N = 500_000
    res = np.zeros((gran + 1, gran + 1))
    samples = [[0,0],[0,0]]
    for i in range(2):
        for j in range(2):
            samples[i][j] = truncated_sample(DISTS[i][j], PARAMS[i][j], rho, N)
    samples=np.array(samples)
    ays,bees = [],[]
    for a in tqdm(range(0, gran + 1)):
        ays.append((a+1)/gran)
        for b in range(0, gran + 1):
            if a == 0: bees.append((2*b+1)/gran)
            #print([a/gran, 0, ((2*b+1)/gran)])
            #print ([(a+1)/gran, 0, 1/((2*b+1)/gran)], rho)
            for i in range(2):
                for j in range(2):
                    if (i == 0 and j == 0) or (i == j and symm): 
                        samples[i][j] = truncated_sample(DISTS[0][0], [(a+1)/gran, 0, 1/((2*b+1)/gran)], rho, N)
                    else: samples[i][j] = truncated_sample(DISTS[i][j], PARAMS[i][j], rho, N)

            #print(samples.shape)
            CIs = []
            for i,transform in enumerate(transforms):
                if i == 0:
                    CIs.append(wsbm_CI_(pi, samples, lambda x : transform[0](x,(a+1)/gran, 1/((2*b+1)/gran)), transform[1]))
                else:
                    CIs.append(wsbm_CI_(pi, samples, transform[0], transform[1]))
            print ((a+1)/gran,((2*b+1)/gran),CIs)

            res[a,b] = np.argmax(CIs)
            #if random.random() < 0.2: res[a,b] = 0 
            #print((a+1)/gran, ((3*b+1)/gran), res[a,b])
            #print ((a+1)/gran,(2*b+1)/gran,res[a,b],CIs)

    x, y = np.meshgrid(bees,ays)
    #z = np.sin(0.5 * x) * np.cos(0.52 * y)

    # Mask various z values.
    #z = np.ma.array(z)

    fig, ax= plt.subplots(ncols=1)
    #for ax in axs:
    cs = ax.pcolormesh(x, y, res, cmap='tab10')
    #ax.contour(cs, c='k')
    ax.set_title(f"Optimal transform for $\\rho = {rho}, \\pi = {pi}$, {dist_label}")

    # Plot grid.
    ax.grid(c='k', alpha=0.3)

    plt.xlabel("$b$")
    plt.ylabel("$a$")
    fig.colorbar(cs)
    plt.show()
    return x,y,res
def plot_contour_beta(model, transforms, rho, gran=30, symm=False):
    pi, DISTS, PARAMS,dist_label = model.pi, model.DISTS, model.PARAMS, model.dist_label
    N = 500_000
    res = np.zeros((gran + 1, gran + 1))
    samples = [[0,0],[0,0]]
    for i in range(2):
        for j in range(2):
            samples[i][j] = truncated_sample(DISTS[i][j], PARAMS[i][j], rho, N)
    samples=np.array(samples)
    ays,bees = [],[]
    for a in tqdm(range(0, gran + 1)):
        ays.append((a+1)/gran)
        for b in range(0, gran + 1):
            if a == 0: bees.append(1+(2*b+1)/gran)
            #print([a/gran, 0, ((2*b+1)/gran)])
            #print ([(a+1)/gran, 0, 1/((2*b+1)/gran)], rho)
            for i in range(2):
                for j in range(2):
                    if (i == 0 and j == 0) or (i == j and symm): 
                        samples[i][j] = truncated_sample(DISTS[0][0], [(a+1)/gran, 1+((2*b+1)/gran)], rho, N)
                    #else: samples[i][j] = truncated_sample(DISTS[i][j], PARAMS[i][j], rho, N)

            #print(samples.shape)
            CIs = []
            for i,transform in enumerate(transforms):
                if i == 0:
                    CIs.append(wsbm_CI_(pi, samples, lambda x : transform[0](x,(a+1)/gran,1+((2*b+1)/gran)), transform[1]))
                else:
                    CIs.append(wsbm_CI_(pi, samples, transform[0], transform[1]))
            res[a,b] = np.argmax(CIs)
            #if random.random() < 0.2: res[a,b] = 0 
            #print((a+1)/gran, ((3*b+1)/gran), res[a,b])
            #print ((a+1)/gran,(2*b+1)/gran,res[a,b],CIs)

    x, y = np.meshgrid(bees,ays)
    #z = np.sin(0.5 * x) * np.cos(0.52 * y)

    # Mask various z values.
    #z = np.ma.array(z)

    fig, ax= plt.subplots(ncols=1)
    #for ax in axs:
    cs = ax.pcolormesh(x, y, res, cmap='tab10')
    #ax.contour(cs, c='k')
    ax.set_title(f"Optimal transform for $\\rho ={rho}, \\pi = {pi}$, {dist_label}")

    # Plot grid.
    ax.grid(c='k', alpha=0.3)

    plt.xlabel("$b$")
    plt.ylabel("$a$")
    fig.colorbar(cs)
    plt.show()
    return x,y,res
#### PLOTTING FUNCTIONS ####


def plot_transforms(PI, R, transforms, labels, mode, dist_label, w = None, gran=20):
    res = np.zeros((gran + 1, len(transforms)))
    
    for rho in tqdm(range(0, gran + 1)):
        if mode == "node":
            X, Y = R[0,1], R[1,0]
            CIs = [
                discretized_chernoff_information(transform(rho/gran, X), transform(rho/gran, Y))
                for transform in transforms
            ]
        elif mode == "edge":
            CR = np.copy(R)
            CIs = []
            for transform in transforms:
                CR = transform(rho/gran, R)
                #for i in range(R.shape[0]):
                #    for j in range(R.shape[1]):
                #        CR[i,j] = transform(rho/gran,R[i,j])
                CIs.append(wsbm_scalar_CI(PI, CR))
        else: assert False, "mode must be one of 'edge' or 'node'"
        res[rho] = CIs

    for i in range(len(transforms)):
        plt.plot([x / gran for x in range(0,gran + 1)], res[:,i], label=labels[i])
    plt.xlabel("w")
    plt.ylabel("chernoff information")
    plt.legend()
    if w is not None and w >= 0 and w <= 1:
        plt.axvline(x = w, color = 'b', linestyle='--')
        plt.xlabel("w")
    plt.title(f"CI against $w$ for $U[0,1]$ vs. {dist_label} ({mode}-tests)")
    plt.show()

def simple_plot_transforms(X, Y, transforms, labels, mode, dist_label, w = None, gran = 20):
    PI = np.diag([0.5,0.5])
    N = len(X)
    R = np.zeros((2,2,N))
    R[0,1] = X
    R[1,0] = Y
    return plot_transforms(PI, R, transforms, labels, mode, dist_label, w, gran)