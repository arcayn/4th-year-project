
from scipy.special import digamma, polygamma
import numpy as np
import scipy

class Transform:
    def __init__(self, f, label):
        self.f = f
        self.label = label

    def apply(self, w, X):
        return self.f(w, X)

#### TRANSFORMS ####
def softmax(X):
    exs = np.exp(X/1000)
    #print (X)
    return exs/np.sum(exs, axis = 1)

def log_dotproduct(X : np.ndarray, Y : np.ndarray):
    assert X.shape[1] == Y.shape[0]
    

def norma(w,X):
    for _ in range(int(w)):
        X = (X @ X.T) / len(X)
    #X = (X - np.mean(X) + 1e-6) / np.sqrt(max(np.var(X), 1e-6))
    #X = np.log(X)
    return np.log(X)
def normn(w,X):
    X = -np.log(X)
    oldX = X
    for _ in range(int(w)):
        X = (X @ oldX) / len(X)
        #X *= np.linalg.norm(oldX, axis = 1) / np.linalg.norm(X, axis = 1)
    #for _ in range(int(w)):
    #    X = (-np.log(X) @ -np.log(X).T @ -np.log(X).T) / (len(X) * len(X))
    return X

def pp_inner(k,n,w,X):
    snk = -np.sum(np.log(X[:,:k]), axis = 1)
    EXP = k * (1 - digamma(k + 1) + digamma(n + 1))
    VAR = k + k * k * (polygamma(1, k + 1) - polygamma(1, n + 1))
    pp = EXP - snk / np.sqrt(VAR)
    return pp

def pp_transform(w, X : np.ndarray):
    #X_ = np.sort(X, axis = 1)
    #k,n = 200, X.shape[1]
    #EXP = (1 - digamma(k + 1) + digamma(n + 1))
    #VAR = k + k * k * (polygamma(1, k + 1) - polygamma(1, n + 1))
    #XX = np.copy(X)
    #for i in range(len(X)):
    #    XX[i][XX[i] >= X_[i][500]] = 1
    return norma(w,X)
    K = X.shape[1]
    outs = np.ones(X.shape[0]) * 100
    for k in range(1, K):
        pp = pp_inner(k,K,w,X)
        outs = np.min((pp, outs), axis = 0)
    #print (outs)
    return np.repeat(outs.reshape(-1,1), X.shape[1], axis = 1)


def pearson_transform(w, x):
    x = x / 1.1
    return np.log(1 - x)

def stouffer_transform(w, x):
    x = x / 1.1
    return scipy.stats.norm.ppf(x)

EdgingtonTransform   = Transform(lambda w,x: x, "Edgington")
FisherTransform      = Transform(lambda w,x : np.log(x), "Fisher")
PearsonTransform     = Transform(pearson_transform, "Pearson")
StoufferTransform    = Transform(stouffer_transform, "Stouffer")
CombinationTransform = Transform(lambda w,x : w * np.log(x) + (1 - w) * x, "F/E combination")
PowerLogTransform    = Transform(pp_transform, "Power log")
PowerTransform       = Transform(normn, "Power Edgington")

TRANSFORMS = [
    EdgingtonTransform,
    FisherTransform,
    #PearsonTransform,
    #StoufferTransform,
    #CombinationTransform,
    PowerLogTransform,
    PowerTransform
]

