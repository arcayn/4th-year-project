import numpy as np
import random
import scipy
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA

def logit(x):
    return 1/(1 + np.exp(-x))

def standard_basis(K, i):
    r = np.zeros(K)
    r[i] = 1
    return r
def S_matrix(C, k, l, t):
    return (1 - t) * np.diag(C[k]) + t * np.diag(C[l])

def standard_dist_matrix(dist, p1, p2 = None):
    if p2 is None: p2 = p1
    return [
        [[dist, scipy.stats.uniform], [scipy.stats.uniform, scipy.stats.uniform]],
        [[p1, []], [[], []]]
    ]
def opposite_dist_matrix(dist, p1, p2 = None):
    if p2 is None: p2 = p1
    return [
        [[scipy.stats.uniform, dist], [dist, scipy.stats.uniform]],
        [[[], p1], [p1, []]]
    ]

def symmetric_dist_matrix(dist, p1, p2 = None):
    if p2 is None: p2 = p1
    return [
        [[dist,scipy.stats.uniform], [scipy.stats.uniform,dist]],
        [[p1,[]], [[],p1]]
    ]

def bipartite_dist_matrix(dist, p1, p2 = None):
    if p2 is None: p2 = p1
    # normal, anomalous, test
    return [
        [
            [scipy.stats.uniform, scipy.stats.uniform, scipy.stats.uniform],
            [scipy.stats.uniform, scipy.stats.uniform, dist],
            [scipy.stats.uniform, dist, scipy.stats.uniform],
        ],
        [
            [[0,1], [0,1], [0,1]],
            [[0,1], [0,1], p1],
            [[0,1], p1, [0,1]],
        ],
    ]

def truncated_sample(dist, params, rho = 0.5, N = 100_000):
    R = dist.rvs(*params, size=2*N)

    R = R[R <= 1]
    R = R[R >= 0]
    while len(R) < N:
        R = np.concatenate((R, dist.rvs(*params, size=N)))
        R = R[R <= 1]
        R = R[R >= 0]
    R = R[:N]
    for k in range(N):
        if random.random() > rho: R[k] = 0.999#random.random() #R[k] = 1
    return R

def matrix_truncated_sample(DISTS, PARAMS, rho = 0.5, N = 100_000):
    K = len(DISTS)
    return np.array(
        [[truncated_sample(DISTS[i][j],PARAMS[i][j], rho, N) for j in range(K)] for i in range(K)],
        dtype="float64"
    )

from scipy.sparse import csgraph
class SpecEmbed:
    def __init__(self, d, laplacian = False):
        self.d = d
        self.laplacian = laplacian

    def fit_transform(self, X, inv = False):
        assert ((X -X.T) < 1e-7).all()
        if self.laplacian:
            X,d = csgraph.laplacian(X, return_diag=True)
            scaling = np.sqrt(abs(d))
            X = (1/scaling) * X * (1/scaling)
            #for i in range(len(X)):
            #    for j in range(i+1, len(X)):
            #        X[i,j] = X[j,i]
        s, U = scipy.linalg.eigh(X)
        Vh = U.T
        #assert ((X - (U @ np.diag(s) @ Vh)) < 1e-5).all()
        #print (s)
        if inv:
            U, s, Vh = U[:, :self.d], s[:self.d], Vh[:self.d, :]
        else:
            U, s, Vh = U[:, -self.d:], s[-self.d:], Vh[-self.d:, :]
        
        #assert ((U - Vh.T) < 1e-5).all()
        return U @ np.sqrt(np.abs(np.diag(s)))

svd_embed = SpecEmbed(d=2)
lsvd_embed = SpecEmbed(d=2,laplacian=True)
#svd_embed = SpectralEmbedding(n_components=2, affinity="precomputed")
pca_embed = PCA(n_components=2)
class WSBMGraph:
    def _gen_WSBM(DISTS, PARAMS, PI, n, rho):
        num_pops = len(PI)
        submat = [int(n*PI[0]),n-int(n*PI[0])]
        assert sum(submat) == n
        graph = None
        for i in range(num_pops):
            row = None
            for j in range(num_pops):
                SM = truncated_sample(DISTS[i][j], PARAMS[i][j], rho, submat[i] * submat[j]).reshape(submat[i], submat[j])
                if row is None: row = SM
                else: row = np.concatenate((row, SM), axis = 1)
            if graph is None: graph = row
            else: graph = np.concatenate((graph, row), axis = 0)
        for i in range(len(graph)):
            for j in range(i, len(graph)):
                graph[j,i] = graph[i,j]
        return graph
    
    def _gp_pair(n):
        orig = list(range(n))
        random.shuffle(orig)
        unperm = [0 for _ in orig]
        for i in range(len(orig)):
            unperm[orig[i]] = i
        return orig,unperm
    
    def __init__(self, DISTS, PARAMS, PI, n, rho):
        self.DISTS, self.PARAMS, self.PI, self.n, self.rho = DISTS, PARAMS, PI, n, rho
        self.graph = WSBMGraph._gen_WSBM(DISTS, PARAMS, PI, n, rho)
        self.graph_stack = []
        self.perm_stack = []

    def transform(self, transform, cache = True):
        if cache: self.graph_stack.append(self.graph)
        self.graph = transform(self.graph)
    
    def untransform(self):
        if len(self.graph_stack) > 0:
            self.graph = self.graph_stack.pop()
            return True
        return False        

    def distribution_statistics(self):
        num_pops = len(self.PI)
        submat = [int(self.n*p) for p in self.PI]
        final_mu = np.zeros((num_pops, num_pops), dtype="float64")
        final_mu2 = np.zeros((num_pops, num_pops),  dtype="float64")
        cum_col = 0
        for i in range(num_pops):
            cum_row = 0
            for j in range(num_pops):
                res = self.graph[cum_col : cum_col + submat[i], cum_row : cum_row + submat[j]]
                cum_row += submat[j]
                temp_mu = np.mean(res)
                final_mu[i][j] = temp_mu
                final_mu2[i][j] = np.mean(np.power(res, 2))
            cum_col += submat[i]
        final_var = final_mu2 - np.power(final_mu, 2)
        return final_mu, final_mu2, final_var
    
    def permute(self, cache = False, return_perm = False):
        perm,unperm = WSBMGraph._gp_pair(len(self.graph))
        self.graph = self.graph[perm]
        self.graph.T[:,:] = self.graph.T[perm]
        assert (self.graph == self.graph.T).all()
        self.perm_stack.append(unperm)
        if return_perm: return unperm, perm
        return unperm
    
    def unpermute(self):
        perm = self.perm_stack.pop()
        self.graph = self.graph[perm]
        self.graph.T[:,:] = self.graph.T[perm]
        assert (self.graph == self.graph.T).all()

    def embed(self, method = "SVD"):
        match method:
            case "SVD":
                return svd_embed.fit_transform(self.graph)
            case "LSVD":
                return lsvd_embed.fit_transform(self.graph)
            case "PCA":
                return pca_embed.fit_transform(self.graph)



class WSBM:
    def __init__(self, dist_label, DISTS, PARAMS, PI):
        self.DISTS, self.PARAMS, self.PI = DISTS, PARAMS, PI
        self.is_symmetric = DISTS[0][0] == DISTS[1][1] and PARAMS[0][0] == PARAMS[1][1]
        self.pi = PI[0]
        self.dist_label = dist_label
        trunc = DISTS[0][0].cdf(1, *PARAMS[0][0]) - DISTS[0][0].cdf(0, *PARAMS[0][0])
        self.alt_dist = lambda x : self.DISTS[0][0].pdf(x, *self.PARAMS[0][0])/trunc

    def sample(self, n, rho):
        return WSBMGraph(self.DISTS, self.PARAMS, self.PI, n, rho)
    
    def sample_permute(self, rho, N, n = 1):
        samples = np.zeros((n, N, N))
        gts = np.zeros((n, N))
        for i in range(n):
            local_graph = WSBMGraph(self.DISTS, self.PARAMS, self.PI, N, rho)
            _,perm = local_graph.permute(False, True)
            samples[i] = local_graph.graph
            gts[i] = np.concatenate((np.ones(int(N*self.pi)), np.zeros(N-int(N*self.pi))))[perm]
        return samples, np.array(gts, dtype="int")
    
    def multi_sample(self, rho, N = 100_000, n = 1_000) -> list[WSBMGraph]:
        it = int(np.ceil(N/np.power((n * np.min(self.PI)), 2)))
        ret = []
        for _ in range(it):
            graph = self.sample(n, rho)
            ret.append(graph)
        return ret
    
    def find_distribution_statistics(self, graphs):
        num_pops = len(self.PI)
        final_mu = np.zeros((num_pops, num_pops), dtype="float64")
        final_mu2 = np.zeros((num_pops, num_pops),  dtype="float64")
        for curr, graph in enumerate(graphs):
            temp_mu, temp_mu2, _ = graph.distribution_statistics()
            final_mu  = (final_mu  * curr + temp_mu )/(curr + 1)
            final_mu2 = (final_mu2 * curr + temp_mu2)/(curr + 1)
        final_var = final_mu2 - np.power(final_mu, 2)
        return final_mu, final_mu2, final_var
