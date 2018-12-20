from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from scipy import linalg
from collections import Counter
import toy_matrices as mat
from AdaptivePELE.analysis import autoCorrelation as autoC
from AdaptivePELE.freeEnergies import runMarkovChainModel as run
from AdaptivePELE.freeEnergies import utils
import time
import matplotlib.pyplot as plt
plt.style.use("ggplot")

C = mat.pre_docking
n = C.shape[0]
# T = run.buildTransitionMatrix(C)
T = utils.buildRevTransitionMatrix(C.astype(np.float))
eigenvals, eigenvectors = linalg.eig(T, left=True, right=False)
sortedIndices = np.argsort(eigenvals)[::-1]
# stationary distribution
goldenStationary = run.getStationaryDistr(eigenvectors[:, sortedIndices[0]])

test_initial = False
run_simulation = True
# test generation of initial structures according to a distribution
numberOfSimulations = 127
possible_state = []
for l in range(n):
    possible_state.extend([l for _ in range(np.round(goldenStationary[l]*numberOfSimulations).astype(int))])
if test_initial:
    nRuns = 1000
    test = "calculated"
    if test == "range":
        title = "Choice over range(n)"
    elif test == "allpossible":
        title = "Choice over list of possibles states"
    elif test == "calculated":
        title = "Calculating frequency"
        initialStates = []
        for i, pi in enumerate(goldenStationary):
            initialStates.extend([i for _ in range(np.round(numberOfSimulations*pi).astype(int))])
        states = len(initialStates)
        if states < numberOfSimulations:
            initialStates.extend(np.random.choice(range(n), size=(numberOfSimulations-states), p=goldenStationary))
        nRuns = 1
    average_prob = np.zeros(n)
    average_divergence = 0
    for ji in range(nRuns):
        if test == "range":
            initialStates = np.random.choice(range(n), size=numberOfSimulations, p=goldenStationary)
        elif test == "allpossible":
            initialStates = np.random.choice(possible_state, size=numberOfSimulations)
        count = Counter(initialStates)
        initial_probs = []
        for i in range(n):
            if i in count:
                initial_probs.append(float(count[i]))
            else:
                initial_probs.append(0.0)
        initial_probs = np.array(initial_probs)
        initial_probs /= initial_probs.sum()
        average_prob += initial_probs
        divergence = run.getRelativeEntropyVectors(goldenStationary, initial_probs)
        # print("Divergence", divergence)
        average_divergence += divergence
    average_divergence /= nRuns
    average_prob /= nRuns
    print("Average divergence over %d runs" % nRuns, average_divergence)
    plt.figure()
    plt.plot(goldenStationary, 'x', label="Probability")
    plt.plot(average_prob, 'o', label="Initial distribution")
    plt.title(title)
    plt.legend()
    plt.savefig("".join(title.split()))

if run_simulation:
    steps = 3000
    taus = np.array(range(1, 2000, 100))
    initial_confs = ["All starting from %d" % i for i in range(n)]
    initial_confs.extend(["Starting from all states", "Starting close to equilibrium"])
    colors = ['xk', 'xr', 'xg', 'xc', 'xb', 'xm', 'ok', 'or', 'og', 'oc', 'ob', 'om']
    assert len(initial_confs) <= len(colors)
    dists = []
    for j in range(len(initial_confs)):
        if j < n:
            initialStates = [j for _ in range(numberOfSimulations)]
        elif j == n:
            initialStates = [i % n for i in range(numberOfSimulations)]
        else:
            initialStates = []
            for i, pi in enumerate(goldenStationary):
                initialStates.extend([i for _ in range(np.round(numberOfSimulations*pi).astype(int))])
            states = len(initialStates)
            if states < numberOfSimulations:
                initialStates.extend(np.random.choice(range(n), size=(numberOfSimulations-states), p=goldenStationary))
            # count = Counter(initialStates)
            # print(goldenStationary)
            # print(count)
        trajs = run.runSetOfSimulations(numberOfSimulations, T, steps, initialStates, verbose=False)
        autoCorr = utils.calculateAutoCorrelation(taus.tolist(), [t.astype(long) for t in trajs], n, len(taus))
        autoC.create_plots(autoCorr, "", False, False, n, taus, title=initial_confs[j])
        allEigen, allProb = run.analyseEigenvalEvol(trajs, taus, n)
        # run.plotEigenvalEvolutionInTau(allEigen, allProb, taus, n, golden=goldenStationary)
        dists.append(run.getRelativeEntropyVectors(goldenStationary, allProb[-1]))
        diffs = [run.getRelativeEntropyVectors(goldenStationary, ij) for ij in allProb]
        # plt.figure()
        # plt.plot(taus, diffs)
        # plt.title(initial_confs[j])
        # plt.xlabel("Lagtime")
        # plt.ylabel("Probability difference")
    plt.figure()
    for l, (d, t) in enumerate(zip(dists, initial_confs)):
        print(l, d, t)
        plt.plot(l, d, colors[l], label=t)
    plt.title("Divergence for the last lagtime with different initial conditions")
    plt.legend()

if test_initial or run_simulation:
    plt.show()
