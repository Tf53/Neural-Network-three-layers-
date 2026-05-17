import numpy as np
from numpy.linalg import inv, norm, solve, det, matrix_rank, cond
from numpy import diag
rng = np.random.default_rng()

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["savefig.bbox"] = "tight"

import warnings
warnings.simplefilter('ignore', FutureWarning)

from typing import Tuple, List, Optional, Union
from tqdm.auto import tqdm





n = 500
A = rng.random(size=(n, n))
print("det(A)", det(A))

n = 600
A = rng.random(size=(n, n))
print("det(A)", det(A))


n = 3000
A = rng.random(size=(n, n))
b = rng.random(n)

x = inv(A) @ b
print("||Ax - b||", norm(A @ x - b))

x = solve(A, b)
print("||Ax - b||", norm(A @ x - b))
