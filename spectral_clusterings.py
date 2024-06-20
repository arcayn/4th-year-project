from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from newbgmm import BayesianGaussianMixture
import numpy as np
import scipy

def spectral(X):
    s, U = scipy.linalg.eigh(X)
    d = 2
    cols,vals = [],[]
    bidx,tidx = 0, len(s) - 1
    for x in range(d):
        if abs(s[bidx]) > abs(s[tidx]):
            cols.append(U.T[bidx])
            vals.append(s[bidx])
            bidx += 1
        else:
            cols.append(U.T[tidx])
            vals.append(s[tidx])
            tidx -= 1

    XA = np.array(cols).T @ np.sqrt(np.abs(np.diag(vals)))
    return XA

def trivial_clustering(embedded, idx = 1):
    res = embedded[:, idx]
    return res - np.min(res)

def GMM_clustering(embedded, bayesian = False, concentration = 0.5):
    if  bayesian:
        gmm = GaussianMixture(n_components=2, random_state=0, max_iter=5_000).fit(embedded)
    else:
        gmm = BayesianGaussianMixture(n_components=2, random_state=0,max_iter=5_000,weight_concentration_prior=concentration).fit(embedded)
    likelihoods = gmm.predict_proba(embedded)

    return [x/y for (x,y) in likelihoods]

def KMeans_clustering(embedded):
    XA = embedded @ embedded.T
    km = KMeans(2).fit(XA)
    return km.predict(XA)

    
def SP_clustering(embedded, pi):
    XA = embedded @ embedded.T
    N = len(XA)
    D = np.zeros(N)
    mu= int(0.5 * pi * N)
    for u in range(N):
        dists = np.sort([np.linalg.norm(XA[u] - XA[v]) for v in range(N)])
        D[u] = dists[mu]
    u_0 = np.argmin(D)
    bound = np.quantile(D, (1 - pi/2))
    u_1,u_1d = 0, 0
    for v in range(N):
        if D[v] > bound: continue
        if D[v] > u_1d:
            u_1 = v
            u_1d = D[v]
    res= [np.linalg.norm(XA[u] - XA[u_0])/np.linalg.norm(XA[u] - XA[u_1]) if u != u_1 else 1e10 for u in range(N)]
    #print(res)
    res = [1 if np.linalg.norm(XA[u] - XA[u_0]) > np.linalg.norm(XA[u] - XA[u_1]) else 0 for u in range(N)]
    return res