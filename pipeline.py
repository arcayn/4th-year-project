import numpy as np
from sampling import WSBM, WSBMGraph
from abc import ABC, abstractmethod
import random
import scipy 

from tqdm import tqdm
from prop import SpectralEmbedding

def identity_transform(X):
    return np.copy(X)

def log_transform(X):
    X = np.copy(X)
    X[X == 0.999] = 1
    return np.log(X)

def stouffer_transform(X):
    X = X/1.01
    res = scipy.stats.norm.ppf(X)
    res[res == scipy.stats.norm.ppf(0.999/1.01)] = 0
    return res

def george_transform(X):
    X = X/1.01
    res = np.log(X) - np.log(1 - X)
    badx = 0.999/1.01
    res[res == np.log(badx/(1-badx))] = 0
    return res

def log_likelihood(X, model):
    trunc = scipy.stats.gamma.cdf(1, ga,0,1/gb) - scipy.stats.gamma.cdf(0, ga,0,1/gb)
    res = scipy.stats.gamma.pdf(X, ga,0,1/gb)/trunc
    res[res == scipy.stats.gamma.pdf(0.999, ga,0,1/gb)] = 1
    return np.log(res)

def threshold(X, tau):
    succ = X < tau
    R = np.zeros(X.shape)
    R[succ] = 1
    return R

def gibbs(A, model, alt_dist, null_dist, alt_prob, m = 100_000, CUTOFF = 20):
    pi = model.pi
    alt_densities = np.log(alt_dist(A))
    alt_densities[alt_densities == np.log(alt_dist(0.999))] = 0
    null_densities = np.log(null_dist(A))
    M0 = np.random.rand(500)#np.concatenate((np.zeros(100), np.ones(400)))
    M1 = 1 - M0
    N = len(A)

    state = np.zeros(N)
    SVD  = SpectralEmbedding(model)
    prior =  SVD.execute(A)
    idxs = np.argpartition(prior, -int(pi *N))[-int(pi * N):]
    state[idxs] = 1
    for iter in tqdm(range(m)):
        for i in range(N):
            state[i] = 1
            res = pi * np.exp(alt_densities[i] @ state)
            state[i] = 1 if random.random() < res/((1 - pi) + res) else 0
    
    mean = np.zeros(N)
    for iter in tqdm(range(250_000)):
        for i in range(N):
            state[i] = 1
            res = pi * np.exp(alt_densities[i] @ state)
            state[i] = 1 if random.random() < res/((1 - pi) + res) else 0
        if iter % 10 == 0:
            nn = iter // 10
            mean = (nn * mean + state) / (nn + 1)

    return mean

import methods
class GibbsSampler(methods.Method):
    def __init__(self, model, m=10_000, CUTOFF=100):
        trunc = model.DISTS[0][0].cdf(1, *model.PARAMS[0][0]) - model.DISTS[0][0].cdf(0, *model.PARAMS[0][0])
        alt_dist = lambda x : model.DISTS[0][0].pdf(x, *model.PARAMS[0][0])/trunc
        null_dist = lambda x : np.ones(x.shape)
        alt_prob= model.pi

        self.model = model
        self.label           = f"GibbsSampler({m})"
        self.transform       = lambda x,s : x
        self.generate_scores = lambda x,s=None : x
        self.infer_missing   = lambda x,s : x
        self.embed           = lambda A,s=None,model=model,alt_dist=alt_dist,null_dist=null_dist,alt_prob=alt_prob,m=m,CUTOFF=CUTOFF : gibbs(A, model, alt_dist, null_dist, alt_prob, m, CUTOFF)
        self.has_inference   = False
        self.default_M       = 0
        self.aux_scores      = lambda x : None
        self.cache           = {}