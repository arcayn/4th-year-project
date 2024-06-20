import plotting
import importlib
importlib.reload(plotting)
import numpy as np
import transforms
import sampling
import matplotlib.pyplot as plt
import sys
import scipy
#np.set_printoptions(suppress=True)
import chernoff
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)
import methods
import prop
importlib.reload(prop)
import pipeline
importlib.reload(pipeline)
import chernoff
importlib.reload(chernoff)
import matplotlib as mpl
import spectral_clusterings
importlib.reload(spectral_clusterings)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'DejaVu Serif'
mpl.rcParams["mathtext.default"] = 'rm'
mpl.rcParams["mathtext.fontset"]= 'dejavuserif'
import pickle

def pred_from_prior(p,gt,pi):
    ll = list(sorted([(y,x) for x,y in enumerate(p)]))
    N = len(p)
    r = np.zeros(N)
    good = sum([gt[x] for _,x in ll[:10]])
    corr = sum(gt)
    nulls = len(gt) - corr
    if good < 6:
        ll = ll[::-1]
    r[[x for _,x in ll[:int(N * pi)]]] = 1
    return r

def hellinger(p,gt):
    #return np.sum(np.abs(p-gt))
    ip,igt = 1-p, 1-gt
    inner,inneri = np.sqrt(p) - gt, np.sqrt(ip) - igt
    return 1/np.sqrt(2) * np.sum(np.sqrt(inner * inner + inneri * inneri))
    assert(p >= 0).all()
    inner = np.sqrt(p) - np.sqrt(gt)
    return (1/np.sqrt(2)) * np.sqrt(np.sum(inner * inner))
        

        

        

def plot_accuracy_pi(algorithms, gran = 30):
    a = 1
    b = 2
    n = 2
    N = 1000
    rho = 0.5

    vals = np.arange(0.1, 0.5, 0.4/gran)
    hells = np.zeros((len(vals), len(algorithms), n))
    accuracies = np.zeros((len(vals), len(algorithms), n))
    mu_1s = []
    for j,pi in enumerate(tqdm(vals)): 
        gamma_dists, gamma_params = sampling.standard_dist_matrix(scipy.stats.gamma, [a,0,1/b])
        model = sampling.WSBM(f"$\\Gamma[{a},{b}]$", gamma_dists, gamma_params, np.array([pi,1 - pi]))
        mu_1s.append(np.mean(sampling.truncated_sample(scipy.stats.gamma, [a,0,1/b], 1, 1_000_000)))

        for fold in range(n):
            local_graph = model.sample(N, rho)
            uperm,perm = local_graph.permute(False, True)
            nanom = int(N * pi)
            gt = np.zeros(N)
            gt[:nanom] = 1
            gt = gt[perm]
            embedded = spectral_clusterings.spectral(local_graph.graph)
            for i,algorithm in enumerate(algorithms):
                p = np.array(algorithm[0](embedded))
                #pred = pred_from_prior(p, gt, 0.15)
                #overlap = sum([1 if gg == pp else 0 for gg,pp in zip(gt,pred)])
                if algorithm[1] == "GMM":
                    psamp = [1 if k > 1 else 0 for k in p[:25]]
                    overlap = sum([1 if gg == pp else 0 for gg,pp in zip(gt[:25],psamp)])
                    if overlap < 13:
                        # H0/H1
                        r_ = p * (1-pi)/pi
                        p = 1/(r_ + 1)
                    else:
                        #H1/H0
                        r_ = p * pi/(1-pi)
                        p = r_/(1 + r_)
                elif algorithm[1] == "K-Means":
                    psamp = [1 if k > 1 else 0 for k in p[:50]]
                    overlap = sum([1 if gg == pp else 0 for gg,pp in zip(gt[:50],psamp)])
                    if overlap < 25:
                        p = 1 - p
                elif algorithm[1] == "Direct eigenvector":
                    nanom = int(N * pi)
                    #print (nanom, p.shape)
                    idxs = np.argpartition(p, -nanom)[-nanom:]
                    overlap = sum(gt[idxs[:25]])
                    if overlap < 13:
                        idxs = np.argpartition(p, nanom)[:nanom]
                    p_ = np.zeros(N)
                    p_[idxs] = 1
                    p = p_
                
                pred = pred_from_prior(p, gt, pi)
                overlap = sum([1 if gg == pp else 0 for gg,pp in zip(gt,pred)])
                hells[j][i][fold] = hellinger(np.array(p),gt)#overlap/N
                accuracies[j][i][fold] = overlap/N
                #print (p, np.sqrt(p))
    acc_mean = np.mean(accuracies, axis = 2)
    acc_std = np.std(accuracies, axis = 2)
    hell_mean = np.mean(hells, axis = 2)
    hell_std = np.std(hells, axis = 2)

    for i in range(len(algorithms)):
        stds = acc_std[:, i]
        yerr = [stds, stds]

        plt.errorbar(vals, acc_mean[:, i], yerr=yerr, capsize=3, fmt="--o", label=algorithms[i][1])
    plt.legend()
    plt.show()

    for i in range(len(algorithms)):
        stds = hell_std[:, i]
        yerr = [stds, stds]

        plt.errorbar(vals, hell_mean[:, i], yerr=yerr, capsize=3, fmt="--o", label=algorithms[i][1])
    plt.legend()
    plt.show()
    return acc_mean, acc_std, hell_mean, hell_std

ALGORITHMS = [
    (spectral_clusterings.trivial_clustering, "Direct eigenvector"),
    (spectral_clusterings.GMM_clustering, "GMM"),
    (spectral_clusterings.KMeans_clustering, "K-Means"),
    ((lambda x : spectral_clusterings.SP_clustering(x, 0.2)), "SP-cluster")
]

RES_GAMMA_VARYING_PI = plot_accuracy_pi(ALGORITHMS, 10)
with open("RES_GAMMA_VARYING_PI.pkl", "wb") as f:
    pickle.dump(RES_GAMMA_VARYING_PI,f)