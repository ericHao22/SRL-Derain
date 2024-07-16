import numpy as np
import cv2

def bfilter2(A, w=5, sigma=(3, 0.1)):
    if A is None or A.size == 0:
        raise ValueError("Input image A is undefined or invalid.")
    if not np.issubdtype(A.dtype, np.float32) or A.shape[-1] not in (1, 3) or np.min(A) < 0 or np.max(A) > 1:
        raise ValueError("Input image A must be a double precision matrix of size NxMx1 or NxMx3 on the closed interval [0,1].")
    
    w = int(np.ceil(w))

    if len(sigma) != 2 or sigma[0] <= 0 or sigma[1] <= 0:
        sigma = [3, 0.1]
    
    if A.shape[-1] == 1:
        B = bfltGray(A.squeeze(-1), w, sigma[0], sigma[1])
    else:
        B = bfltColor(A, w, sigma[0], sigma[1])
    
    return B

def bfltGray(A, w, sigma_d, sigma_r):
    X, Y = np.meshgrid(np.arange(-w, w + 1), np.arange(-w, w + 1))
    G = np.exp(-(X**2 + Y**2) / (2 * sigma_d**2))
    
    dim = A.shape[:2]
    B = np.zeros_like(A)
    
    for i in range(dim[0]):
        for j in range(dim[1]):
            iMin = max(i - w, 0)
            iMax = min(i + w + 1, dim[0])
            jMin = max(j - w, 0)
            jMax = min(j + w + 1, dim[1])
            I = A[iMin:iMax, jMin:jMax]
            
            H = np.exp(-(I - A[i, j])**2 / (2 * sigma_r**2))
            
            F = H * G[iMin - i + w:iMax - i + w, jMin - j + w:jMax - j + w]
            B[i, j] = np.sum(F * I) / np.sum(F)
    
    return B

def bfltColor(A, w, sigma_d, sigma_r):
    X, Y = np.meshgrid(np.arange(-w, w + 1), np.arange(-w, w + 1))
    G = np.exp(-(X**2 + Y**2) / (2 * sigma_d**2))
    
    A_lab = cv2.cvtColor(A, cv2.COLOR_BGR2Lab)
    
    dim = A_lab.shape[:2]
    B_lab = np.zeros_like(A_lab)
    
    sigma_r *= 100
    
    for i in range(dim[0]):
        for j in range(dim[1]):
            iMin = max(i - w, 0)
            iMax = min(i + w + 1, dim[0])
            jMin = max(j - w, 0)
            jMax = min(j + w + 1, dim[1])
            I = A_lab[iMin:iMax, jMin:jMax]
            
            dL = I[:, :, 0] - A_lab[i, j, 0]
            da = I[:, :, 1] - A_lab[i, j, 1]
            db = I[:, :, 2] - A_lab[i, j, 2]
            H = np.exp(-(dL**2 + da**2 + db**2) / (2 * sigma_r**2))
            
            F = H * G[iMin - i + w:iMax - i + w, jMin - j + w:jMax - j + w]
            norm_F = np.sum(F)
            B_lab[i, j, 0] = np.sum(F * I[:, :, 0]) / norm_F
            B_lab[i, j, 1] = np.sum(F * I[:, :, 1]) / norm_F
            B_lab[i, j, 2] = np.sum(F * I[:, :, 2]) / norm_F
    
    B_rgb = cv2.cvtColor(B_lab, cv2.COLOR_Lab2BGR)
    
    return B_rgb
