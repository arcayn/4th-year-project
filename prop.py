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

def simplified_belief_prop(A, model, alt_dist, null_dist, alt_prob, N = 10):
    # NOT WORKING
    pi = model.pi
    alt_densities = alt_dist(A)
    null_densities = null_dist(A)
    marginal_densities = alt_prob * alt_densities + (1 - alt_prob) * null_densities
    M1 = np.ones(len(A)) * pi
    M0 = np.ones(len(A)) * (1-pi)
    for _ in tqdm(range(N)):
        nu1 = np.ones(len(A))
        for i in range(len(A)):
            for j in range(len(A)):
                if A[i][j] == 0.999: continue
                nu1[i] *= pi / marginal_densities[i][j]
                nu1[i] *= alt_densities[i][j] * pi * M1[j] + null_densities[i][j] * (1 - pi) * M0[j]
        nu0 = np.ones(len(A))
        for i in range(len(A)):
            for j in range(len(A)):
                if A[i][j] == 0.999: continue
                nu0[i] *= (1 - pi) / marginal_densities[i][j]
                nu0[i] *= null_densities[i][j] * pi * M1[j] + null_densities[i][j] * (1 - pi) * M0[j]
        print (nu1[:10], nu0[:10])
        M1 = nu1 / (nu1 + nu0)
        print (M1[:10])
        M0 = nu0 / (nu1 + nu0)      
    return M1

def belief_prop(A, model, alt_dist, null_dist, alt_prob, N = 10):
    pi = model.pi
    alt_densities = alt_dist(A)
    null_densities = null_dist(A)
    marginal_densities = alt_prob * alt_densities + (1 - alt_prob) * null_densities
    M1 = np.ones((len(A), len(A))) * pi
    M0 = np.ones((len(A), len(A))) * (1-pi)
    for _ in tqdm(range(N)):
        nu1 = np.ones((len(A), len(A)))
        nu0 = np.ones((len(A), len(A)))
        for i in range(len(A)):
            for j in range(len(A)):
                # this is the message which node i sends to its neighbour j
                if A[i][j] == 0.999: continue
                mult = 1#pi / marginal_densities[i][j]
                mult *= alt_densities[i][j] * pi * M1[j][j] + null_densities[i][j] * (1 - pi) * M0[j][j]
                for k in range(len(A)):
                    if k != j: nu1[i][k] *= mult
                mult = 1#(1-pi) / marginal_densities[i][j]
                mult *= null_densities[i][j] * pi * M1[j][j] + null_densities[i][j] * (1 - pi) * M0[j][j]
                for k in range(len(A)):
                    if k != j: nu0[i][k] *= mult
    FIN1 = np.ones(len(A))
    FIN0 = np.ones(len(A))
    for i in range(len(A)):
        for j in range(len(A)):
            # this is the message which node i sends to its neighbour j
            if A[i][j] == 0.999: continue
            mult = pi / marginal_densities[i][j]
            mult *= alt_densities[i][j] * pi * M1[j][j] + null_densities[i][j] * (1 - pi) * M0[j][j]

            mult2 = (1-pi) / marginal_densities[i][j]
            mult2 *= null_densities[i][j] * pi * M1[j][j] + null_densities[i][j] * (1 - pi) * M0[j][j]
            FIN1[i] *= mult2/max(1e-20,mult)
    
        if FIN1[i] > np.median(FIN1) * 2: FIN1[i] = np.median(FIN1) * 2
    return FIN1


def variant_belief_prop_embed(A, model, alt_dist, null_dist, alt_prob, n = 2, CUTOFF = 20):
    pi = model.pi
    alt_densities = alt_dist(A)
    null_densities = null_dist(A)
    M = np.ones(len(A)) * pi
    joint0 = np.ones(len(A))
    joint1 = np.ones(len(A))
    factors = np.ones((len(A),len(A)))
    for _ in tqdm(range(n)):
        joint1 = np.ones(len(A))
        for j in range(len(A)):
            for i in range(len(A)):
                if A[i][j] == 0.999: continue
                CM = M[i]
                cfactor = (1 - CM) + alt_densities[i][j] * CM
                joint1[j] *= cfactor
                factors[j][i] = cfactor
            M[j] = joint1[j] * pi / (joint1[j] * pi + joint0[j] * (1 - pi))
    ret = joint0 / joint1


    tg = A
    mu = np.mean(A)

    cov = np.zeros((len(tg), len(tg)))
    for i in tqdm(range(len(tg))):
        for j in range(len(tg)):
            cnt = 0
            for k in range(len(tg)):
                if (A[i][k] == 0.999 or A[j][k] == 0.999): continue
                cov[i][j] += (A[i][k] - mu) * (A[j][k] - mu)
                cnt += 1
            if cnt > 0: cov[i][j] /= cnt
                
    vars = np.std(tg, axis = 0)
    varm = np.zeros((len(tg),len(tg)))
    for i in (range(len(tg))):
        for j in range(len(A)):
            varm = vars[i] * vars[j]
    cor = cov / varm - np.eye(len(tg))
    
    for i in range(len(A)):
        if ret[i] > CUTOFF: ret[i] = CUTOFF
    
    final = np.zeros((len(ret), 2))
    for i in range(len(A)):
        final[i][0] = ret[i]
        final[i][1] = ret[np.argmax(cor[i])]
    return final


def variant_belief_prop(A, model, alt_dist, null_dist, alt_prob, m = 2, CUTOFF = 20):
    pi = model.pi
    alt_densities = alt_dist(A)
    null_densities = null_dist(A)
    M = np.ones(len(A)) * pi
    joint0 = np.ones(len(A))
    joint1 = np.ones(len(A))

    joint1_ = np.ones(len(A))
    factors = np.ones((len(A),len(A)))
    for iter in tqdm(range(m)):
        joint1 = np.ones(len(A))
        M_ = np.copy(M)
        for j in range(len(A)):
            for i in range(len(A)):
                if A[i][j] == 0.999 or i == j: continue
                CM = M[i]
                #if iter > 0: 
                #    cj = (M[i] * (joint1_[i] * pi + joint0[i] * (1 - pi))) / pi
                #    cj /= factors[i][j]
                #    CM = cj# * pi / (cj * pi + joint0[j] * (1 - pi))
                cfactor = (1-CM)+ alt_densities[i][j] * CM
                joint1[j] = joint1[j] * cfactor
                #factors[j][i] = cfactor
            M_[j] = joint1[j] * pi / (joint1[j] * pi + joint0[j] * (1 - pi))
        M = M_
        #joint1_ = np.copy(joint1)
    ret = joint0 / joint1
    CUTOFF = np.sort(ret)[-int(len(A) * 0.01)]
    for i in range(len(ret)):
        if ret[i] > CUTOFF: ret[i] = CUTOFF
    return ret

LR = 1
def mean_field_variational(A, model, alt_dist, null_dist, alt_prob, m = 2, CUTOFF = 20):
    pi = model.pi
    alt_densities = A#alt_dist(A)#np.log(alt_dist(A))
    null_densities = np.log(null_dist(A))
    M0 = np.random.rand(len(A))
    M1 = 1 - M0
    N = len(A)

    for iter in tqdm(range(m)):
        M0_ = np.copy(M0)
        M1_ = np.copy(M1)

        for i in range(len(M1_)):
            r0 = np.sum(0 * alt_densities[i])
            r1 = np.sum(1 * alt_densities[i])
            #mid = (r0+r1)/2
            #r0 -= mid; r1-= mid
            f0 = (1-pi)/np.exp(r0)
            f1 = r1#np.exp(r1)
            #M0[i] = f0/(f0 + f1)
            M1[i] = f1
        #M0,M1=M0_,M1_
        elbo = 0
        for j in range(N):
            for k in range(j, N):
                elbo += (M1[j] * M1[k] + M0[j] * M0[k]) * alt_densities[j][k]
            elbo += M0[j] * np.log((1 - pi)/M0[j]) + M1[j] * np.log(pi/M1[j])
        if iter % 50 == 0:
            print (iter, elbo)
            print(M0[:10], M0[-10:])
    #M0 /= (M0 + M1); M1 /= (M0 + M1)
    M = M0
    return M1
    #print ((M1 / np.min(M1))[:10],(M1 / np.min(M1))[-10:])
    if (M > 0.99999).all(): return M1 / np.min(M1)
    return M / np.min(M)
    R = np.zeros(N)
    for i in range(N):
        if M1[i] > M0[i]: R[i] = 1
    return R

def mean_field_variational(A, model, alt_dist, null_dist, alt_prob, m = 2, CUTOFF = 20):
    pi = model.pi
    alt_densities = np.log(alt_dist(A))
    null_densities = np.log(null_dist(A))
    M0 = np.random.rand(len(A))
    M1 = 1 - M0
    N = len(A)

    for iter in tqdm(range(m)):
        M0_ = np.copy(M0)
        M1_ = np.copy(M1)

        for i in range(len(M1_)):
            r0 = np.sum(M0 * alt_densities[i])
            r1 = np.sum(M1 * alt_densities[i])
            #mid = (r0+r1)/2
            #r0 -= mid; r1-= mid
            f0 = (1-pi)*np.exp(r0)
            f1 = pi*np.exp(r1)
            M0[i] = f0/(f0 + f1)
            M1[i] = f1/(f0 + f1)
        #M0,M1=M0_,M1_
        elbo = 0
        for j in range(N):
            for k in range(j, N):
                elbo += (M1[j] * M1[k] + M0[j] * M0[k]) * alt_densities[j][k]
            elbo += M0[j] * np.log((1 - pi)/M0[j]) + M1[j] * np.log(pi/M1[j])
        if False and iter % 50 == 0:
            print (iter, elbo)
            print(M0[:10], M0[-10:])
    #M0 /= (M0 + M1); M1 /= (M0 + M1)
    M = M0
    return M1
    #print ((M1 / np.min(M1))[:10],(M1 / np.min(M1))[-10:])
    if (M > 0.99999).all(): return M1 / np.min(M1)
    return M / np.min(M)
    R = np.zeros(N)
    for i in range(N):
        if M1[i] > M0[i]: R[i] = 1
    return R


def mean_field_variational(A, model, alt_dist, null_dist, alt_prob, m = 2, CUTOFF = 20):
    pi = model.pi
    alt_densities = np.log(alt_dist(A))
    null_densities = np.log(null_dist(A))
    M0 = np.random.rand(500)#np.concatenate((np.zeros(100), np.ones(400)))
    M1 = 1 - M0
    N = len(A)

    for iter in tqdm(range(m)):
        M0_ = np.copy(M0)
        M1_ = np.copy(M1)

        ress = 0
        for j in range(N):
            for k in range(j, N):
                ress += (M1[j] * M1[k] + M0[j] * M0[k]) * alt_densities[j][k]

        for i in range(len(M1_)):
            o0,o1 = M0[i], M1[i]

            #M0[i], M1[i] = 1., 0.
            ress0 = ress
            ress0 += np.log(1 - pi)
            for j in range(N):
                ress0 -= (M1[j] * M1[i] + M0[j] * M0[i]) * alt_densities[j][i]
                ress0 += M0[j] * alt_densities[j][i]

            #M0[i], M1[i] = 0., 1.
            ress1 = ress
            ress1 += np.log(pi)
            for j in range(N):
                ress1 -= (M1[j] * M1[i] + M0[j] * M0[i]) * alt_densities[j][i]
                ress1 += M1[j] * alt_densities[j][i]

            mid = (ress1 + ress0)/2

            ress1 -= mid; ress0 -= mid
            f0,f1 = np.exp(ress0), np.exp(ress1)
            M0[i] = f0/(f0 + f1)
            M1[i] = f1/(f0 + f1)
            #print (i,ress0,ress1,M0_[i],M1_[i])
            #M0[i], M1[i] = o0,o1
        #M0,M1=M0_,M1_

        elbo = 0
        for j in range(N):
            for k in range(j, N):
                elbo += (M1[j] * M1[k] + M0[j] * M0[k]) * alt_densities[j][k]
            elbo -= M0[j] * np.log((1 - pi)/M0[j]) + M1[j] * np.log(pi/M1[j])
        #print (iter, elbo)
    #R = np.zeros(N)
    #for i in range(N):
    #    if M1[i] > M0[i]: R[i] = 1
    #return R
    #M0 /= (M0 + M1); M1 /= (M0 + M1)
    M = M0
    #print(M0,M1)
    if (M > 0.99999).all(): return M1
    return M

def unweighted_mean_field_variational(A, model, alt_dist, null_dist, alt_prob, m = 2, CUTOFF = 20):
    pi = model.pi
    null_p,alt_p = model.PARAMS[0][1][0],model.PARAMS[0][0][0]
    null_densities = null_dist(A)
    M0 = np.random.rand(len(A))
    M1 = 1 - M0
    N = len(A)

    for iter in tqdm(range(m)):
        M0_ = np.copy(M0)
        M1_ = np.copy(M1)
        for i in range(N):
            first = (A[i] * np.log(null_p / (1 - null_p)) * M1) + (A[i] * np.log(alt_p / (1 - alt_p)) * M0)
            fsum = np.sum(first) - first[i]
            second = (N - 1) * (1 - pi) * np.log(1 - alt_p) + N * pi * np.log(1 - null_p)
            f0 = (1 - pi) * np.exp(fsum + second)

            first = (A[i] * np.log(null_p / (1 - null_p)) * M0) + (A[i] * np.log(alt_p / (1 - alt_p)) * M1)
            fsum = np.sum(first) - first[i]
            second = (N - 1) * pi * np.log(1 - alt_p) + N * (1-pi) * np.log(1 - null_p)
            f1 = pi * np.exp(fsum + second)

            M0[i] = f0/(f0 + f1)
            M1[i] = f1/(f0 + f1)
        #M0,M1=M0_,M1_
        
        elbo = 0
        for j in range(N):
            for k in range(j, N):
                elbo += (M1[j] * M1[k] + M0[j] * M0[k]) * (A[j][k] * np.log(alt_p/(1 - alt_p)) + np.log(1 - alt_p)) 
                elbo += (M1[j] * M0[k] + M0[j] * M1[k]) * (A[j][k] * np.log(null_p/(1 - null_p)) + np.log(1 - null_p)) 
            elbo += M0[j] * np.log((1 - pi)/M0[j]) + M1[j] * np.log(pi/M1[j])
        #print (iter, elbo)
        #print(M0,M1)
    #M0 /= (M0 + M1); M1 /= (M0 + M1)
    M = M0
    #print(M0,M1)
    if (M > 0.99999).all(): return M1
    return M
    R = np.zeros(N)
    for i in range(N):
        if M1[i] > M0[i]: R[i] = 1
    return R




class VariantBeliefProp(methods.Method):
    def __init__(self, model, m=2, CUTOFF=100):
        alt_dist = lambda x : model.DISTS[0][0].pdf(x, *model.PARAMS[0][0])
        null_dist = lambda x : np.ones(x.shape)
        alt_prob= model.pi

        self.model = model
        self.label           = f"VariantBeliefProp({m},{CUTOFF})"
        self.transform       = lambda x,s : x
        self.generate_scores = lambda x,s=None : x
        self.infer_missing   = lambda x,s : x
        self.embed           = lambda A,s=None,model=model,alt_dist=alt_dist,null_dist=null_dist,alt_prob=alt_prob,m=m,CUTOFF=CUTOFF : variant_belief_prop(A, model, alt_dist, null_dist, alt_prob, m, CUTOFF)
        self.has_inference   = False
        self.default_M       = 0
        self.aux_scores      = lambda x : None
        self.cache           = {}
class MeanFieldVariational(methods.Method):
    def __init__(self, model, m=2, CUTOFF=100):
        alt_dist = lambda x : model.DISTS[0][0].pdf(x, *model.PARAMS[0][0])
        null_dist = lambda x : np.ones(x.shape)
        alt_prob= model.pi

        self.model = model
        self.label           = f"MeanFieldVariational({m})"
        self.transform       = lambda x,s : x
        self.generate_scores = lambda x,s=None : x
        self.infer_missing   = lambda x,s : x
        self.embed           = lambda A,s=None,model=model,alt_dist=alt_dist,null_dist=null_dist,alt_prob=alt_prob,m=m,CUTOFF=CUTOFF : mean_field_variational(A, model, alt_dist, null_dist, alt_prob, m, CUTOFF)
        self.has_inference   = False
        self.default_M       = 0
        self.aux_scores      = lambda x : None
        self.cache           = {}
class UnweightedMeanFieldVariational(methods.Method):
    def __init__(self, model, m=2, CUTOFF=100):
        #alt_dist = lambda x : model.DISTS[0][0].pdf(x, *model.PARAMS[0][0])
        null_dist = lambda x : np.ones(x.shape)
        alt_prob= model.pi

        self.model = model
        self.label           = f"UnweightedMeanFieldVariational({m})"
        self.transform       = lambda x,s : x
        self.generate_scores = lambda x,s=None : x
        self.infer_missing   = lambda x,s : x
        self.embed           = lambda A,s=None,model=model,alt_dist=0,null_dist=null_dist,alt_prob=alt_prob,m=m,CUTOFF=CUTOFF : unweighted_mean_field_variational(A, model, alt_dist, null_dist, alt_prob, m, CUTOFF)
        self.has_inference   = False
        self.default_M       = 0
        self.aux_scores      = lambda x : None
        self.cache           = {}

class DiagEmbed(methods.Method):
    def __init__(self, model, m=2, CUTOFF=100):
        alt_dist = lambda x : model.DISTS[0][0].pdf(x, *model.PARAMS[0][0])
        null_dist = lambda x : np.ones(x.shape)
        alt_prob= model.pi

        def do_embed(A,s=None):
            A1 = np.float64(A < 0.999)
            t1 = np.sum(A1, axis = 0)
            print(np.mean(t1),np.std(t1))
            
            scores = np.zeros(len(A))
            A_ = A[A != 0.999]
            mu,sigma = np.mean(A_), np.var(A_)
            for thresh in tqdm([0.01,0.05,0.1]):
                A2 = np.float64(A < thresh)
                t2 = np.sum(A2, axis = 0)
                for i in range(len(A)):
                    scores[i] += 1 if t1[i] == 0 else t2[i]/t1[i]
            return scores

        self.model = model
        self.label           = f"DiagEmbed"
        self.transform       = lambda x,s : x
        #self.generate_scores = lambda x,s=None : x
        #self.infer_missing   = lambda x,s : x
        self.embed           = do_embed
        self.has_inference   = False
        self.default_M       = 0
        self.aux_scores      = lambda x : None
        self.cache           = {}


def spectral_embed(tg,inv):
    svd_1d = sampling.SpecEmbed(d=2)
    sums = svd_1d.fit_transform(tg, inv)
    gmm = GaussianMixture(n_components=2, random_state=0).fit(sums)
    likelihoods = gmm.predict_proba(sums)
    #return sums[:,1]
    return [x/y for (x,y) in likelihoods]

import spectral_clusterings
importlib.reload(spectral_clusterings)
class SpectralEmbedding(methods.Method):
    def __init__(self, model, inv=False, norm=False):

        self.model = model
        self.label           = f"SpectralEmbed({'hi' if inv else 'lo'}, log-centered)"
        def do_transform(x,s=None):
            orig = x
            x = np.log(x)
            #x -= np.mean(x)
            x[orig == 0.999] = 0
            return x
        self.transform       = do_transform
        self.generate_scores = lambda x,s=None : x
        self.infer_missing   = lambda x,s : x
        def do_embed(A,s=None,inv=inv,norm=norm):
            em = spectral_clusterings.spectral(A)
            #return spectral_clusterings.GMM_clustering(em)
            #return spectral_clusterings.KMeans_clustering(em)
            #return spectral_clusterings.SP_clustering(em, 0.2)
            return spectral_clusterings.trivial_clustering(em, 0)
            if norm: em = 1 / 1 + np.exp(-em)
            return em
        self.embed           = do_embed
        self.has_inference   = False
        self.default_M       = 0
        self.aux_scores      = lambda x : None
        self.cache           = {}

class SpectralEmbeddingNoLog(SpectralEmbedding):
    def __init__(self, model, inv=False, norm=False):
        SpectralEmbedding.__init__(self, model, inv, norm)
        self.label           = f"SpectralEmbed({'hi' if inv else 'lo'}, centered)"
        def do_transform(x,s=None):
            x = x - np.mean(x)
            return x
        self.transform       = do_transform

class SpectralEmbeddingNoCenter(SpectralEmbedding):
    def __init__(self, model, inv=False, norm=False):
        SpectralEmbedding.__init__(self, model, inv, norm)
        self.label           = f"SpectralEmbed({'hi' if inv else 'lo'}, log-uncentered)"
        def do_transform(x,s=None):
            x = -np.log(x)
            return x
        self.transform       = do_transform

class SpectralEmbeddingInfer(SpectralEmbedding):
    def __init__(self, model, inference, inference_label, scoring = None, scoring_label = None, inv=False, norm=False):
        SpectralEmbedding.__init__(self, model, inv, norm)
        self.label           = f"SpectralEmbed({inference_label}/{scoring_label}, {'hi' if inv else 'lo'})"
        if scoring is not None:
            self.generate_scores = scoring
        self.infer_missing   = inference
        self.has_inference   = True
        self.default_M       = 1
        def do_transform(x,s=None):
            x = -np.log(x)
            x -= np.mean(x)
            return x
        self.transform       = do_transform

def top_avg(tg, i, j, m = 5):
    topsi = np.sort(tg[i][tg[i] != 0.999])[:m]
    topsj = np.sort(tg[j][tg[j] != 0.999])[:m]
    return np.mean(np.concatenate((topsi, topsj)))

class SpectralEmbeddingInferAvg(SpectralEmbeddingInfer):
    def __init__(self, model, top = 5, scoring = None, scoring_label = None, inv=True, norm=False):
        def do_infer(A, s = None):
            x = np.copy(A)
            for i in range(len(x)):
                for j in range(len(x)):
                    if A[i][j] == 0.999:
                        if A[j][i] != 0.999: x[i][j] = x[j][i]
                        else:
                            x[i][j] = x[j][i] = top_avg(A,i,j,top)
            return x

        SpectralEmbeddingInfer.__init__(self, model, do_infer, f"Top{top}Average", scoring, scoring_label, inv, norm)
