"""
baselines/test.py
=================
CLI smoke-test for :func:`SPOT_GreedySubsetSelection`.

Generates a random cost matrix and a random target marginal on GPU (via CuPy)
and runs the greedy SPOT selection algorithm to verify correctness and timing.

Usage
-----
    python test.py <numY> <numX> <m>

Arguments
---------
numY : int
    Number of source points (rows of the cost matrix).
numX : int
    Number of target points (columns of the cost matrix).
m : int
    Number of prototypes to select. Must satisfy ``m < numY``.

Example
-------
    python test.py 1000 1000 80
"""

from SPOTgreedy import SPOT_GreedySubsetSelection
import cupy as cp
from cupy import random
import sys


def main():
    if len(sys.argv) != 4:
        sys.exit(
            "Please specify 3 arguments: numY, numX, m.\n"
            "Example: python test.py 1000 1000 80"
        )

    numY = int(sys.argv[1])
    numX = int(sys.argv[2])
    m    = int(sys.argv[3])

    if m >= numY:
        sys.exit("No. of prototypes (m) must be strictly less than the source size (numY).")

    # Generate a random integer cost matrix and target marginal on GPU.
    C = random.randint(100, size=(numY, numX))
    t = random.randint(100, size=numX)

    results = SPOT_GreedySubsetSelection(C, t, m)
    print("Prototype indices:", results)


if __name__ == "__main__":
    main()

