import numpy as np
from scipy.linalg import sqrtm


def subspace_distance(A1, A2):
    # A1, A2: (d, k) matrix
    k = A1.shape[1]
    return 1 - np.trace(A2.T @ A1 @ np.linalg.inv(A1.T @ A1) @ A1.T @ A2 @ np.linalg.inv(A2.T @ A2)) / k


def rotate(U, V, start, end):
    U_ = U[:, start:end]
    V_ = V[:, start:end]
    Vhat_ = V_ @ np.linalg.inv(sqrtm(V_.T @ V_))
    Vrot_ = Vhat_ @ (Vhat_.T @ U_)
    return Vrot_


def procrustes(A, Ahat, start, end):
    # A (target), Ahat (learned)
    # k = end - start
    A_ = A[:, start:end]
    Ahat_ = Ahat[:, start:end]

    U, S, Vt = np.linalg.svd(Ahat_.T @ A_)
    Q = U @ Vt  # optimal orthogonal transformation
    # return Q
    return Ahat_ @ Q  # (d, k)
