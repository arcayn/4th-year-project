from tqdm import tqdm
import matplotlib.pyplot as plt
from transforms import *
from sampling import *
from chernoff import *

def plot_CIs(res, model, transforms, w = None):
    gran = len(res) - 1
    for i in range(len(transforms)):
        plt.plot([x / gran for x in range(0,gran + 1)], res[:,i], label=transforms[i].label)
    plt.xlabel("w")
    plt.ylabel("chernoff information")
    plt.legend()
    if w is not None and w >= 0 and w <= 1:
        plt.axvline(x = w, color = 'b', linestyle='--')
        plt.xlabel("w")
    plt.title(f"CI against $w$ for $U[0,1]$ vs. {model.dist_label} (edge-tests)")
    plt.show()

def plot_embed_scatter(X,Y, title):
    plt.scatter(X[:, 0], X[:,1], c = "blue")
    plt.scatter(Y[:, 0], Y[:,1], c = "red")
    plt.title(title)
    plt.show()

def plot_transforms_analytic(
        model: WSBM, transforms: list[Transform], rho: float,
        show_stats = True, show_CIs = True,
        N = 100_000, n = 4_000, w = None, gran=2):
    
    res = np.zeros((gran + 1, len(transforms)))
    graphs = model.multi_sample(rho, N, n)
    for p in tqdm(range(0, gran + 1)):
        CIs = []
        for i,transform in enumerate(transforms):
            for graph in graphs: graph.transform(lambda x : transform.apply(p/gran, x))
            B,_,C = model.find_distribution_statistics(graphs)
            if show_stats: print (transform.label,B,C)
            for graph in graphs: graph.untransform()
            if show_CIs: CIs.append(wsbm_chernoff_information_(np.diag(model.PI), B, C)[0,1]) 
        if show_CIs: res[p] = CIs

    if show_CIs: plot_CIs(res, model, transforms, w)

def plot_transforms_empirical(
        model, transforms, rho,
        show_scatter = True, show_CIs = True, mode = "SVD",
        N = 100_000, n = 1_000, gran=2):
    
    res = np.zeros((gran + 1, len(transforms)))
    graphs = model.multi_sample(rho, N, n)
    unperms = [graph.permute() for graph in graphs]
    submat = [int(n*p) for p in model.PI]
    for p in tqdm(range(0, gran + 1)):
        CIs = []
        for i,transform in enumerate(transforms):
            X,Y = None,None
            for graph,unperm in zip(graphs,unperms):
                graph.transform(lambda x : transform.apply(p/gran, x))
                graph_test = graph.embed(mode)[unperm]
                if X is None: X = graph_test[:submat[0]]
                else: X = np.concatenate((X, graph_test[:submat[0]]), axis = 1)
                if Y is None: Y = graph_test[submat[0]:]
                else: Y = np.concatenate((Y, graph_test[submat[0]:]), axis = 1)
                graph.untransform()
            if show_scatter: plot_embed_scatter(X, Y, f"$U[0,1]$ vs. {model.dist_label}, transformed by {transform.label}, embedded by {mode}")
            if show_CIs:
                dci = d2_discretized_chernoff_information(
                    X,Y, 5
                )
                print (dci)
                CIs.append(dci)
        if show_CIs: res[p] = CIs

    if show_CIs: plot_CIs(res, model, transforms)


import matplotlib.pyplot as plt

from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay, roc_curve,roc_auc_score




def ROC_plot(y_true, y_pred, label):
    """ each input should be split into folds """
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    if len(y_true.shape) == 1: y_true, y_pred = y_true.reshape(1, -1), y_pred.reshape(1, -1)
    for i, (y_true_fold, y_pred_fold) in enumerate(zip(y_true, y_pred)):
        tauc =  roc_auc_score(y_true_fold, y_pred_fold)
        if tauc < 0.5: 
            idxs0, idxs1 = y_true_fold == 0, y_true_fold == 1
            y_true_fold[idxs0] = 1.0
            y_true_fold[idxs1] = 0.0
        viz = RocCurveDisplay.from_predictions(
            y_true_fold,
            y_pred_fold,
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=f"ROC: {label}",
    )
    ax.legend(loc="lower right")
    plt.show()

def ROC_surface_data(y_true_tensor, y_pred_tensor):
    """ each input should be split into granularities, then folds """
    mean_fpr = np.linspace(0, 1, 100)

    if len(y_true_tensor.shape) == 1: y_true_tensor, y_pred_tensor = y_true_tensor.reshape(1, -1), y_pred_tensor.reshape(1, -1)
    if len(y_true_tensor.shape) == 2: y_true_tensor, y_pred_tensor = y_true_tensor.reshape(1, y_true_tensor.shape[0], -1), y_pred_tensor.reshape(1, y_pred_tensor.shape[0], -1)
    granularities = []
    for gran, (y_true, y_pred) in enumerate(zip(y_true_tensor, y_pred_tensor)):
        tprs = []
        for i, (y_true_fold, y_pred_fold) in enumerate(zip(y_true, y_pred)):
            tauc =  roc_auc_score(y_true_fold, y_pred_fold)
            if tauc < 0.5: 
                idxs0, idxs1 = y_true_fold == 0, y_true_fold == 1
                y_true_fold[idxs0] = 1.0
                y_true_fold[idxs1] = 0.0
            fpr, tpr, _ = roc_curve(
                y_true_fold,
                y_pred_fold
            )
            roc_auc_score(y_true_fold, y_pred_fold)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        granularities.append((mean_fpr[1:], mean_tpr[1:], mean_auc))

    # Make data.
    X = np.array([i / len(y_true_tensor) for i in range(1,len(y_true_tensor)+1)])
    Y = np.array([g[0] for g in granularities])
    X = np.array(np.meshgrid(X, Y[0]))[0].T
    Z = np.array([g[1] for g in granularities])

    return X,Y,Z
from matplotlib.colors import CenteredNorm
def plot_ROC_surface(X,Y,Z,label,fig=None,ax=None,cmap="Viridis"):
    if fig is None or ax is None: fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


    print (X.shape, Y.shape, Z.shape)
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, norm=CenteredNorm(0),
                        linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set(
        title=f"ROC: {label}"
    )