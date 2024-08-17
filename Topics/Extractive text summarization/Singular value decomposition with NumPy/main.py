import numpy as np

def solution(mtx:list) -> float:
    U, S, Vh = np.linalg.svd(mtx, full_matrices=True)

    return round(np.sum(S), 1)