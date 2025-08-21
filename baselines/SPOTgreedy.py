import numpy as np
import scipy
from scipy import sparse
import time

def SPOT_GreedySubsetSelection(C, targetMarginal, m):
    # Assumes one source point selected at a time, which simplifies the code.
    # C: Cost matrix of OT: number of source x number of target points {[numY * numX]}
    # targetMarginal: 1 x number of target (row-vector) size histogram of target distribution. Non negative entries summing to 1 {[1*numX]}
    # m: number of prototypes to be selected.

    targetMarginal = targetMarginal / np.sum(targetMarginal)
    numY = C.shape[0]
    numX = C.shape[1]
        # Fix targetMarginal shape if needed
    if targetMarginal.size != numX:
        print(f"Warning: targetMarginal size {targetMarginal.size} does not match numX {numX}. Auto-fixing.")
        targetMarginal = np.ones(numX) / numX
    print(f"Number of sources (Y): {numY}, Number of targets (X): {numX}")
    allY = np.arange(numY)
    # just to make sure we have a row vector.
    targetMarginal = targetMarginal.reshape(1, numX)

    # Intialization
    S = np.zeros((1, m), dtype=int)
    timeTaken = np.zeros((1, m), dtype=int)
    setValues = np.zeros((1, m), dtype=int)
    sizeS = 0
    currOptw = []
    currMinCostValues = np.ones((1, numX)) * 1000000
    currMinSourceIndex = np.zeros((1, numX), dtype=int)
    remainingElements = allY.copy()
    chosenElements = []
    iterNum = 0
    start = time.time()
    while sizeS < m:
        iterNum += 1
        remainingElements = remainingElements[~np.in1d(remainingElements, np.array(chosenElements))]
        temp1 = np.maximum(currMinCostValues - C, 0)
        temp1 = np.matmul(temp1, targetMarginal.T)
        incrementValues = temp1[remainingElements]
        maxIncrementIndex = np.argmax(incrementValues)
        # Choosing the best element
        chosenElement = remainingElements[maxIncrementIndex]
        chosenElements.append(chosenElement)
        S[0][sizeS] = chosenElement
        # Update currMinCostValues and currMinSourceIndex vectors
        tempIndex = (currMinCostValues - C[chosenElement, :]) > 0
        D = C[chosenElement]
        currMinCostValues[0, tempIndex[0]] = D[tempIndex[0]]
        currMinSourceIndex[0, tempIndex[0]] = sizeS
        # Current objective and other booking
        currObjectiveValue = np.sum(np.dot(currMinCostValues, targetMarginal.T))
        setValues[0][sizeS] = currObjectiveValue
        if sizeS == m-1:
            print("targetMarginal", targetMarginal)
            gammaOpt = sparse.csr_matrix((targetMarginal[0], (currMinSourceIndex[0], range(0, numX))), shape=(m, numX))
            print("gammaOpt \n", gammaOpt)
            currOptw = np.sum(gammaOpt, axis=1).flatten()
            print("currOptw \n", currOptw)
        sizeS += 1
    end = time.time()
    print("S : ", S)
    print("Time : ", end - start)
    return S[0]