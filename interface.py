import numpy as np
from sampling import WSBM, WSBMGraph
from abc import ABC, abstractmethod
import random
import scipy
from tqdm import tqdm
from prop import SpectralEmbedding
import spectral_clusterings
import refinements
import sampling

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
    res = model.alt_dist(X)
    res[X == 0.999] = 1
    return np.log(res)

def threshold(X, tau):
    succ = X < tau
    R = np.zeros(X.shape)
    R[succ] = 1
    return R

class SpectralPipeline:
    def __init__(self, spec, model):
        self.init = spec["init"]
        self.model = model
        if "threshold" in spec:
            self.threshold = spec["threshold"]
        else:
            assert spec["init"] != "threshold", "A threshold must be supplied for threshold transform!"

        self.cluster = spec["cluster"]
        self.refiner = spec["refine"]

    def do_transform(self, A):
        match self.init:
            case "stouffer":
                return stouffer_transform(A)
            case "george":
                return george_transform(A)
            case "likelihood":
                return log_likelihood(A, self.model)
            case "fisher":
                return log_transform(A)
            case "threshold":
                return threshold(A, self.threshold)
            case _:
                return np.copy(A)
            
    def do_spec_cluster(self, G):
        embedded = spectral_clusterings.spectral(G)
        N = len(G)
        pi = self.model.pi
        match self.cluster:
            case "gmm":
                ratios = np.array(spectral_clusterings.GMM_clustering(embedded))
                thresholded = np.zeros(N)
                thresholded[ratios > 1] = 1
                # here we are assuming that pi < 0.5
                if sum(thresholded) > 0.5 * N:
                    # H0/H1
                    r_ = ratios * (1-pi)/pi
                    return 1/(r_ + 1)
                else:
                    #H1/H0
                    r_ = ratios * pi/(1-pi)
                    return r_/(1 + r_)
            case "sp":
                initial = spectral_clusterings.SP_clustering(embedded, self.model.pi)
                if sum(initial) > 0.5 * N:
                    return 1 - initial
                else:
                    return initial
            case "kmeans":
                initial = spectral_clusterings.KMeans_clustering(embedded)
                if sum(initial) > 0.5 * N:
                    return 1 - initial
                else:
                    return initial
            case _:
                return spectral_clusterings.trivial_clustering(embedded)
    
    def do_refine(self, A, prior, max_iter = None, warm_up = None):
        match self.refiner:
            case "gibbs":
                if warm_up is None: warm_up = 50_000
                if max_iter is None: max_iter = 100_000
                return refinements.gibbs(A, prior, self.model, warm_up, max_iter, self.model.is_symmetric)
            case "icm":
                if warm_up is None: warm_up = 250
                if max_iter is None: max_iter = 500
                return refinements.ICM(A, prior, self.model, warm_up, max_iter, self.model.is_symmetric)
            case "bp":
                warm_up = 0
                if max_iter is None: max_iter = 1_000
                return refinements.BP(A, prior, self.model, warm_up, max_iter, self.model.is_symmetric)
            case "sp":
                if warm_up is None: warm_up = 100
                if max_iter is None: max_iter = 100
                return refinements.SP_refine(A, prior, self.model, warm_up, max_iter, self.model.is_symmetric)

    def predict_(self, A, max_iter = None, warm_up = None):
        G = self.do_transform(A)
        prior = self.do_spec_cluster(G)
        return self.do_refine(A, prior, max_iter, warm_up)
    
    def predict(self, A, max_iter = None, warm_up = None):
        if len(A.shape) == 2:
            return self.predict_(A, max_iter, warm_up)
        
        preds = np.zeros((A.shape[0], A.shape[1])) 
        for i in range(len(A)):
            preds[i] = self.predict_(A[i], max_iter, warm_up)
        return preds

def build_model(alt_dist, alt_params, pi, symmetric, label = ""):
    if symmetric: dists, params = sampling.symmetric_dist_matrix(alt_dist, alt_params)
    else: dists, params = sampling.standard_dist_matrix(alt_dist, alt_params)
    return sampling.WSBM(label, dists, params, np.array([pi,1 - pi]))
