from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from scipy import linalg
from collections import Counter
from AdaptivePELE.freeEnergies import runMarkovChainModel as run
from AdaptivePELE.freeEnergies import utils
import matplotlib.pyplot as plt
plt.style.use("ggplot")

C = np.array([
    [2, 1, 0, 0, 0, 0],
    [1, 2, 1, 0, 0, 0],
    [0, 1, 2, 1, 0, 0],
    [0, 0, 1, 2, 1, 0],
    [0, 0, 0, 1, 2, 1],
    [0, 0, 0, 0, 1, 2]
])
# C = np.array([
#     [100, 1, 0, 0, 0, 0],
#     [1, 100, 1, 0, 0, 0],
#     [0, 1, 100, 1, 0, 0],
#     [0, 0, 1, 100, 1, 0],
#     [0, 0, 0, 1, 100, 1],
#     [0, 0, 0, 0, 1, 100]
# ])
# C = np.array([
#     [100, 10, 0, 0, 0, 0],
#     [1, 100, 10, 0, 0, 0],
#     [0, 1, 100, 10, 0, 0],
#     [0, 0, 1, 100, 10, 0],
#     [0, 0, 0, 1, 100, 10],
#     [0, 0, 0, 0, 1, 100]
# ])
n = C.shape[0]
# T = run.buildTransitionMatrix(C)
T = utils.buildRevTransitionMatrix(C.astype(np.float))
eigenvals, eigenvectors = linalg.eig(T, left=True, right=False)
sortedIndices = np.argsort(eigenvals)[::-1]
# stationary distribution
goldenStationary = run.getStationaryDistr(eigenvectors[:, sortedIndices[0]])

test_initial = True
run_simulation = False
# test generation of initial structures according to a distribution
numberOfSimulations = 100
possible_state = []
for l in range(n):
    possible_state.extend([l for _ in range(int(goldenStationary[l]*numberOfSimulations))])
if test_initial:
    average_prob = np.zeros(n)
    average_divergence = 0
    for ji in range(10):
        # initialStates = np.random.choice(range(n), size=numberOfSimulations, p=goldenStationary)
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
        print("Divergence", divergence)
        average_divergence += divergence
    average_divergence /= 10.0
    average_prob /= 10.0
    print("Average divergence over 10 runs", average_divergence)
    plt.figure()
    plt.plot(goldenStationary, 'x', label="Probability")
    plt.plot(average_prob, 'o', label="Initial distribution")
    plt.legend()

if run_simulation:
    steps = 3000
    taus = np.array(range(1, 200, 10))
    dists = []
    for j in range(3):
        if j == 0:
            initialStates = [0 for _ in range(numberOfSimulations)]
            initial_conf = "All starting from 0"
        elif j == 1:
            initialStates = [i % n for i in range(numberOfSimulations)]
            initial_conf = "Starting from all states"
        else:
            initialStates = np.random.choice(range(n), size=numberOfSimulations, p=goldenStationary)
            initial_conf = "Starting close to equilibrium"
        trajs = run.runSetOfSimulations(numberOfSimulations, T, steps, initialStates, verbose=False)
        allEigen, allProb = run.analyseEigenvalEvol(trajs, taus, n)
        # run.plotEigenvalEvolutionInTau(allEigen, allProb, taus, n, golden=goldenStationary)
        dists.append(run.getRelativeEntropyVectors(goldenStationary, allProb[-1]))
        diffs = [run.getRelativeEntropyVectors(goldenStationary, ij) for ij in allProb]
        plt.figure()
        plt.plot(taus, diffs)
        plt.title(initial_conf)
        plt.xlabel("Lagtime")
        plt.ylabel("Probability difference")
    plt.figure()
    plt.plot(dists)

if test_initial or run_simulation:
    plt.show()
