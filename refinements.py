import numpy as np
from sampling import WSBM, WSBMGraph
from abc import ABC, abstractmethod
import random
#from tqdm import tqdm

tqdm = lambda x : x
def sample_from_prior(prior, pi):
    N = len(prior)
    state = np.zeros(N)
    idxs = np.argpartition(prior, -int(pi *N))[-int(pi * N):]
    state[idxs] = 1
    return state

def gibbs(A, prior, model, warm_up = 100_000, avg = 250_000, symm = False):
    pi = model.pi
    alt_densities = model.alt_dist(A)
    alt_densities = np.log(alt_densities)
    alt_densities[abs(alt_densities) > 100] = 100
    alt_densities[A == 0.999] = 0
    N = len(prior)

    state = sample_from_prior(prior, pi)
    mean = np.zeros(N)
    for iter in tqdm(range(warm_up + avg)):
        for i in range(N):
            if symm:
                state[i] = 1
                res1 = pi * np.exp(alt_densities[i] @ (state == state[i]))
                state[i] = 0
                res0 = (1-pi) * np.exp(alt_densities[i] @ (state == state[i]))
            else:
                state[i] = 1
                res1 = pi * np.exp(alt_densities[i] @ state)
                res0 = 1 - pi
            #print (res0,res1)
            state[i] = 1 if random.random() < res1/(res0 + res1) else 0
            if iter >= warm_up and (iter - warm_up) % 10 == 0:
                nn = (iter - warm_up) // 10
                mean = (nn * mean + state) / (nn + 1)

    return mean

def SP_refine(A, prior, model, warm_up = 100, avg = 100, symm = False):
    pi = model.pi
    alt_densities = np.log(model.alt_dist(A))
    alt_densities[alt_densities == np.log(model.alt_dist(0.999))] = 0
    N = len(prior)

    state = sample_from_prior(prior, pi)
    mean = np.zeros(N)
    for iter in tqdm(range(warm_up + avg)):
        newstate = np.zeros(len(state))
        for i in range(N):
            if symm:
                state[i] = 1
                res1 = (state == state[i]) @ alt_densities[i]
                state[i] = 0
                res0 = 0(state == state[i]) @ alt_densities[i]
                newstate[i] = 1 if res1 > res0 else 0
            else:
                newstate[i] = 1 if state @ alt_densities[i] > 0 else 0
        state = newstate
        if iter >= warm_up:
            nn = (iter - warm_up)
            mean = (nn * mean + state) / (nn + 1)

    return mean

def CAVI(A, prior, model, warm_up = 0, m = 25, symm = False):
    M = np.copy(prior)
    N = len(A)
    pi = model.pi
    alt_densities = np.log(model.alt_dist(A))
    alt_densities[A == 0.999] = 0
    old_elbo = 0
    elbo_cnt = 0
    for iter in tqdm(range(m)):
        for i in range(N):
            h0 = (1 - M) @ alt_densities[i] if symm else 0
            h1 = M @ alt_densities[i]
            e0 = (1-pi) * np.exp(h0)
            e1 = pi * np.exp(h1)
            M[i] = e1/(e0 + e1)
        if iter % 10 == 0:
            elbo = 0
            for j in range(N):
                for k in range(j, N):
                    elbo += (M[j] * M[k] + ((1-M[j]) * (1-M[k]) if symm else 0)) * alt_densities[j][k]
                elbo += (1-M[j]) * np.log((1 - pi)/(1-M[j])) + M[j] * np.log(pi/M[j])
            if abs(old_elbo - elbo) < 1e-3: elbo_cnt += 1
            else: elbo_cnt = 0
            old_elbo = elbo
            if elbo_cnt >= 3: break
            #print(M0[:10], M0[-10:])
    return M

def BP(A, prior, model, warm_up = 0, m = 25, symm = False):
    M = np.copy(prior)
    N = len(A)
    pi = model.pi
    alt_densities = model.alt_dist(A)
    alt_densities[A == 0.999] = 0
    #alt_densities[alt_densities == 1] = 0.999
    
    #messages = np.random.random((N,N))
    messages = np.repeat(M.reshape(1, -1), N, axis = 0)
    old_elbo = 0
    elbo_cnt = 0
    for iter in tqdm(range(m)):
        for i in range(N):        
            initial = messages[:,i] * alt_densities[i] + (1 - messages[:,i])
            initial[A[i] == 0.999] = 1
            initial[initial == 0] = 1e-8
            marg1 = pi * np.prod(initial)
            mess1 = marg1 / initial

            if symm:
                initial0 = (1-messages[:,i]) * alt_densities[i] + messages[:,i]
                initial0[A[i] == 0.999] = 1
                initial0[initial0 == 0] = 1e-8
                marg0 = (1-pi) * np.prod(initial0)
                mess0 = marg0 / initial0
            else:
                mess0 = np.ones(N) * (1 - pi)
                marg0 = 1-pi
            #initials = np.repeat(initial.reshape(1, -1), N, axis = 0)
            #np.fill_diagonal(initials,1)
            #mess1 = pi * np.prod(initials, axis = 1)
                
            messages[i] = mess1 / (mess1 + mess0)
            M[i] = marg1 / (marg1 + marg0)
        if iter % 10 == 0:
            elbo = 0
            for j in range(N):
                for k in range(j, N):
                    elbo += M[j] * M[k] * alt_densities[j][k]
                component_1 = M[j]
                component_2 = (1-M[j])
                if abs(component_1) < 1e-5: component_1 = 1e-5
                if abs(component_2) < 1e-5: component_2 = 1e-5
                elbo += (1-M[j]) * np.log((1 - pi)/component_2) + M[j] * np.log(pi/component_1)
            if abs(old_elbo - elbo) < 1e-3: elbo_cnt += 1
            else: elbo_cnt = 0
            old_elbo = elbo
            if elbo_cnt >= 3: break
            #print (iter, elbo)
            #print(M0[:10], M0[-10:])

    return M

def compute_hamiltonian(alt_densities, pi, state, symm):
    N = len(state)
    if symm:
        state = np.copy(state)
        state[state==0] = -1
        deltas = state.reshape(N, 1) @ state.reshape(N, 1).T
        deltas += 1
    else:
        deltas = state.reshape(N, 1) @ state.reshape(N, 1).T
    pairwise = np.sum(deltas * alt_densities)
    external = np.sum([np.log(pi) if s == 1 else np.log(1 - pi) for s in state])
    return -(pairwise + external)


def ICM(A, prior, model, warm_up = 0, m = 25, symm = False):
    pi = model.pi
    alt_densities = np.log(model.alt_dist(A))
    alt_densities[alt_densities == np.log(model.alt_dist(0.999))] = 0
    N = len(A)
    T_min, T_max = 0.1, 1.1
    temps = np.arange(T_min, T_max, 0.1)
    N_T = len(temps)
    J = np.std(A)
    print(J)
    replicas = np.array([sample_from_prior(prior, pi) for _ in range(N_T * 2)])
    mean, nn = np.zeros(N), 0
    for iter in tqdm(range(warm_up + m)):

        # gibbs sweeps - alternatively do M-H here

        for i,replica in enumerate(replicas):
            temp = temps[i//2]
            for v in range(N):
                if symm:
                    replica[i] = 1
                    res1 = pi * np.exp(alt_densities[i] @ (replica == replica[i]))
                    replica[i] = 0
                    res0 = (1-pi) * np.exp(alt_densities[i] @ (replica == replica[i]))
                else:
                    replica[i] = 1
                    res1 = pi * np.exp(alt_densities[i] @ replica)
                    res0 = 1 - pi
                #print (replica.shape, alt_densities.shape)
                res1 = np.pow(pi * res1, 1/temp)
                res0 = np.pow((1-pi) * res0, 1/temp)
                replica[v] = 1 if random.random() < res1/(res1 + res0) else 0

        if iter > warm_up:
            tidx = int((1 - T_min)/0.1)
            state = (replicas[2*tidx] + replicas[2*tidx + 1])/2
            mean = (nn * mean + state) / (nn + 1)
            nn += 1

        # HCMs
                
        for i,replica in enumerate(replicas):
            temp = temps[i//2]
            if temp <= J and i%2 == 0:
                overlap = replicas[i] == replicas[i+1]
                idxs = [j for j in range(N) if not overlap[j]]
                if len(idxs) == 0: continue

                flip_idx = random.sample(idxs, 1)[0]
                complete = []
                horizon = [flip_idx]
                while len(horizon) > 0:
                    curr = horizon.pop()
                    complete.append(curr)
                    for u in idxs:
                        if u in complete or u in horizon: continue
                        if A[u][curr] != 0.999: horizon.append(u)
                for idx in complete:
                    replicas[i][idx] = 1 - replicas[i][idx]
                    replicas[i+1][idx] = 1 - replicas[i+1][idx]

        # tempering

        for i in range(1,N_T):
            idxlow = random.randint(0,1)
            idxhigh = random.randint(0,1)
            ham_low = compute_hamiltonian(alt_densities, pi, replicas[(i-1) * 2 + idxlow], symm)
            ham_hi = compute_hamiltonian(alt_densities, pi, replicas[i * 2 + idxhigh], symm)
            swap = random.random() < np.exp((temps[i] - temps[i-1]) * (ham_hi - ham_low))
            if swap:
                old = replicas[(i-1) * 2 + idxlow]
                replicas[(i-1) * 2 + idxlow] = replicas[i * 2 + idxhigh]
                replicas[i * 2 + idxhigh] = old
                #print ("tempering")

    return mean

def SW(A, prior, model, warm_up = 0, m = 25):
    pi = model.pi
    alt_densities = np.log(model.alt_dist(A))
    alt_densities_ = model.alt_dist(A)
    alt_densities[A == 0.999] = 0
    N = len(A)
    T_min, T_max = 0.1, 1.1
    temps = np.arange(T_min, T_max, 0.1)
    N_T = len(temps)
    J = np.std(A)
    print(J)
    state = sample_from_prior(prior, pi)
    mean, nn = np.zeros(N), 0
    for iter in tqdm(range(warm_up + m)):
        D = np.zeros((N,N))
        D_attr = np.zeros(N)
        for i in range(N):
            for j in range(i,N):
                delta = state[i] * state[j]
                if alt_densities[i][j] < 0:
                    if delta != 1 and random.random() < (1 - alt_densities_[i][j]):
                        D[i][j] = D[j][i] = 1
                else:
                    if delta == 1 and random.random() < (1 - 1/alt_densities_[i][j]) :
                        D[i][j] = D[j][i] = 1
                        D_attr[i]= 1;D_attr[j] = 1

        if iter > warm_up:
            tidx = int((1 - T_min)/0.1)
            mean = (nn * mean + state) / (nn + 1)
            nn += 1

        # sweeps - alternatively do M-H here
                        
        to_visit = set(list(range(N)))
        preassigned = set()
        while len(to_visit) > 0:
            horizon = [to_visit.pop()]
            while len(horizon) > 0:
                curr = horizon.pop()
                if curr in to_visit: to_visit.remove(curr)
                if curr not in preassigned:
                    if D_attr[curr] == 1: state[curr] = 1
                    else:
                        state[curr] = 1
                        res = pi * np.exp(alt_densities[curr] @ state)
                        state[curr] = 1 if random.random() < res/((1 - pi) + res) else 0
                for u in to_visit:
                    if u in horizon: continue
                    if D[u][curr] == 1:
                        if state[curr] == 1:
                            preassigned.add(u)
                            if alt_densities[u][curr] < 0: state[curr] = 0
                            else: state[curr] = 1
                        horizon.append(u)
    return mean