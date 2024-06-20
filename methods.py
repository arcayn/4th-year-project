import plotting
import importlib
importlib.reload(plotting)
import numpy as np
import transforms
import sampling
import matplotlib.pyplot as plt
import sys
#np.set_printoptions(suppress=True)
import chernoff
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)

class Method:
    def __init__(self, label, transform, embed, infer_missing = None, generate_scores = None, aux_scores = None, M = 0):
        self.label           = label
        self.transform       = transform
        self.generate_scores = embed if generate_scores is None else generate_scores
        self.infer_missing   = infer_missing
        self.embed           = embed
        self.has_inference   = infer_missing is not None
        self.default_M       = M if self.has_inference else 0
        self.aux_scores      = self.generate_scores if aux_scores is None else aux_scores
        self.cache           = {}

    def execute(self, X, M = None):
        if M is None: M = self.default_M
        for _ in range(M):
            scores = self.generate_scores(X)
            X = self.infer_missing(X, scores)
           
        scores = self.aux_scores(X)
        X = self.transform(X, scores)
        return self.embed(X, scores)

    def clear_cache(self, e = None):
        if e is None: self.cache = {}
        else: self.cache.pop(e, None)
    
    def gen_folds(self, model, rho, N, n = 3, norm = True):
        R = np.zeros((n, N))
        for fold in range(n):
            local_graph = model.sample(N, rho)
            uperm = local_graph.permute()
            R[fold] = self.execute(local_graph.graph)
            R[fold] = R[fold][uperm]
            if norm:
                R[fold] -= np.min(R[fold])
                if np.max(R[fold]) != 0: R[fold] /= np.max(R[fold])
        return R
    
    def make_tag(self, model, N, n_gran, norm):
        return f"{model.dist_label}_{N}_{n_gran}_{norm}"

    def request_cache(self, n, model, N, n_gran, norm):
        tag = self.make_tag(model, N, n_gran, norm)
        if tag in self.cache:
            l,v = self.cache[tag]
            if l >= n: return n, v[:, :n, :]
            else: return l, v
        return 0, None
    
    def insert_cache(self, X, model, N, n_gran, norm):
        tag = self.make_tag(model, N, n_gran, norm)
        if tag in self.cache:
            l,v = self.cache[tag]
            self.cache[tag] = (l + len(X), np.concatenate((v, X), axis = 1))
        else:
            self.cache[tag] = (len(X), X)

    def gen_folds_granular(self, model, N, n_gran = 10, n = 3, norm = True):
        l,v = self.request_cache(n, model, N, n_gran, norm)
        if l == n: return v
        else:
            R = np.zeros((n_gran, n - l, N))
            for gran in range(1, n_gran + 1):
                R[gran - 1] = self.gen_folds(model, gran/n_gran, N, n - l, norm)
            self.insert_cache(R, model, N, n_gran, norm)
            return self.request_cache(n,model, N, n_gran, norm)[1]
    
    def test_hist(self, model, rho, N = 500, norm = True, M = None):
        local_graph = model.sample(N, rho)
        uperm = local_graph.permute()
        sums = self.execute(local_graph.graph)
        sums = sums[uperm]
        pi = model.pi
        delt = np.max(sums) - np.min(sums)
        print (f"Min Score: {np.min(sums)}, Max Score: {np.max(sums)}, Delta: {delt}")
        if delt == 0: 
            print ("Plot failed, zero score delta")
            return
        #print (sums.shape)
        print ("Chernoff information:", chernoff.discretized_chernoff_information(sums[:int(pi * N)], sums[int(pi * N):]))
        
        bins = np.arange(np.min(sums), np.max(sums) + delt/39, delt/40)
        plt.hist(sums[:int(pi * N)], bins)
        plt.show()
        plt.hist(sums[int(pi * N):], bins)
        plt.show()

    def test_matshow(self, model, rho, N = 500, norm = True):
        sums = self.gen_folds(model, rho, N, 1, norm)[0]
        pi = model.pi
        delt = np.max(sums) - np.min(sums)
        print (f"Min Score: {np.min(sums)}, Max Score: {np.max(sums)}, Delta: {delt}")
        print ("Chernoff information:", chernoff.discretized_chernoff_information(sums[:int(pi * N)], sums[int(pi * N):]))
        plt.matshow(sums.reshape(-1, 20))
        plt.colorbar()

    def test_roc(self, model, rho, N = 500, n = 3, norm = True):
        sums = self.gen_folds(model, rho, N, n, norm)
        print (n,sums.shape)
        pi = model.pi
        true_0 = np.concatenate((np.zeros(int(N * pi)), np.ones(N - int(N * pi)))).reshape(1, -1)
        plotting.ROC_plot(np.repeat(true_0,n,axis=0), sums, f"{self.label}, {model.dist_label}")

def compare_methods(model, method1, method2, N = 500, n_gran = 10, n = 3, norm = True):
    SUMS1 = method1.gen_folds_granular(model, N, n_gran, n, norm)
    SUMS2 = method2.gen_folds_granular(model, N, n_gran, n, norm)

    pi = model.pi
    true_0 = np.concatenate((np.zeros(int(N * pi)), np.ones(N - int(N * pi)))).reshape(1, -1)
    true_1 = np.concatenate((np.ones(int(N * pi)), np.zeros(N - int(N * pi)))).reshape(1, -1)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X1,Y1,Z1=plotting.ROC_surface_data(np.repeat(np.repeat(true_0,3,axis=0).reshape(1,3,-1),10,axis=0), SUMS1)
    X2,Y2,Z2=plotting.ROC_surface_data(np.repeat(np.repeat(true_0,3,axis=0).reshape(1,3,-1),10,axis=0), SUMS2)

    #ax.plot_surface(X1,Y1,Y1*0)
    ax.set(
        xlabel="edge probability", ylabel="false positive rate", zlabel="true positive rate"
    )
    plotting.plot_ROC_surface(X1,Y1,Z1 - Z2, f"Advantage of {method1.label} over {method2.label}, {model.dist_label}",fig,ax,plotting.cm.get_cmap('bwr_r'))
    plt.show()
