from math import sqrt
import numpy as np
import scipy

import matplotlib.pyplot as plt

import ot

def ParseImageHist(filename, N=32):
    A = plt.imread(filename)
    n, m, _ = A.shape
    s = n//N
    A = A[0:n:s, 0:m:s, 0].astype(float)/255
    print(A.shape)
    return A

def ShowImage(A):
    fig, ax = plt.subplots()
    plt.imshow(A)
    ax.autoscale()
    ax.set_aspect('equal', 'box')
    plt.axis('off')
    plt.show()

def MakeOTproblem(A, B, p=1):
    # Suppose square matrix of length 1 (point a positions 1/n)
    n, _ = A.shape
    N = n*n

    print('n:', n, 'N:', N)
    a = A.flatten()
    b = B.flatten()

    C = np.zeros((N, N))
    for i in range(n):
        for j in range(n):
            for v in range(n):
                for w in range(n):
                    if p == 1:
                        C[i*n+j, v*n+w] = np.abs(i-v) + np.abs(j-w)
                    elif p == 2:
                        C[i*n+j, v*n+w] = (i-v)**2 + (j-w)**2
    return a, b, C


def LogSinkhorn(a, b, C, eps=0.1, maxit=100):
    n = len(a)
    m = len(b)

    return np.zeros((n,m))

def NaiveSinkhorn(a, b, C, eps=0.1, maxit=100):
    n = len(a)
    m = len(b)

    return np.zeros((n,m))

if __name__ == "__main__":
    Test = 6

    if Test == 1:
        # Basic test
        a = np.array([1.0, 1.0])
        x = np.array([0, 1])

        b = np.array([1.0, 1.0])
        y = np.array([1, 2])

        # Compute the cost matrix
        C = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                C[i, j] = pow(np.linalg.norm(x[i] - y[j]), 2)
    
        #P = ot.sinkhorn(a, b, C, reg=0.1, verbose=True)
        #print(P)

        NaiveSinkhorn(a, b, C, eps=0.1, maxit=1000)

    elif Test == 2:
        # Basic test
        a = np.array([1.0, 1.0])
        x = np.array([(0, 0), (1, 1)])
        
        b = np.array([1.0, 1.0])
        y = np.array([(0, 1), (0, 1)])

        a = a/np.sum(a)
        b = b/np.sum(b)

        # Compute the cost matrix
        # reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        C = scipy.spatial.distance.cdist(x, y)
        
        gamma = 0.005
        fobj, P = NaiveSinkhorn(a, b, C, eps=gamma, maxit=1000)
        print(np.sum(P), P,  np.sum(C * P) + gamma * np.sum(P * np.log(P)- P))
       
        P = ot.sinkhorn(a, b, C, reg=gamma, verbose=True)
        print(np.sum(P), P,  np.sum(C * P) + gamma * np.sum(P * np.log(P)-P))

        P = ot.lp.emd(a, b, C)
        print(np.sum(P), P, np.sum(C * P))

    elif Test == 3:
        # Hexagon test
        a = np.array([1.0, 1.0, 1.0])
        x = np.array([(1, 0), (-1/2, sqrt(3)/2), (-1/2, -sqrt(3)/2)])
        
        b = np.array([1.0, 1.0, 1.0])
        y = np.array([(1/2, sqrt(3)/2), (-1, 0), (1/2, -sqrt(3)/2)])

        a = a/np.sum(a)
        b = b/np.sum(b)

        C = scipy.spatial.distance.cdist(x, y)

        gamma = 0.1
        fobj, P = LogSinkhorn(a, b, C, eps=gamma, maxit=1000)
        print(np.sum(P), P,  np.sum(C * P) + gamma * np.sum(P * np.log(P)- P))

        P, ll = ot.sinkhorn(a, b, C, reg=gamma, verbose=True, log=True)
        print(ll)
        print(np.sum(P), P,  np.sum(C * P) + gamma * np.sum(P * np.log(P)-P))

        P = ot.lp.emd(a, b, C)
        print(np.sum(P), P, np.sum(C * P))

    elif Test == 4:
        # Semi-Hexagon test
        a = np.array([1.0, 1.0, 1.0])
        x = np.array([(1, 0), (-1/2, sqrt(3)/2), (-1/2, -sqrt(3)/2)])
        
        b = np.array([1.0, 1.0])
        y = np.array([(1/2, sqrt(3)/2), (-1, 0)])

        a = a/np.sum(a)
        b = b/np.sum(b)

        C = scipy.spatial.distance.cdist(x, y)

        gamma = 1
        fobj, P = LogSinkhorn(a, b, C, eps=gamma, maxit=1000)
        print(np.sum(P), P,  np.sum(C * P) + gamma * np.sum(P * np.log(P)- P))

        P, ll = ot.sinkhorn(a, b, C, reg=gamma, verbose=True, log=True)
        print(ll)
        print(np.sum(P), P,  np.sum(C * P) + gamma * np.sum(P * np.log(P)-P))

        P = ot.lp.emd(a, b, C)
        print(np.sum(P), P, np.sum(C * P))


    elif Test == 5:
        A = ParseImageHist('../data/dotmark/picture32_1004.png')
        B = ParseImageHist('../data/dotmark/picture32_1006.png')
        #ShowImage(A)
        #ShowImage(B)
        
        a, b, C = MakeOTproblem(A, B, p=2)

        gamma = 0.005

        fobj, P = NaiveSinkhorn(a, b, C, eps=gamma, maxit=1000)
        print(np.sum(P), np.sum(C * P), np.sum(C * P) + gamma * np.sum(P * np.log(P)-P))

        P = ot.sinkhorn(a, b, C, reg=gamma, verbose=True, numItermax=10000, method='sinkhorn')
        #ShowImage(P)
        print(np.sum(P), np.sum(C * P), np.sum(C * P) + gamma * np.sum(P * np.log(P)-P))

        P = ot.lp.emd(a, b, C)
        print(np.sum(P), np.sum(C * P))
        #ShowImage(P)
        
    elif Test == 6:
        A = ParseImageHist('../data/dotmark/picture64_1004.png', N=64)
        B = ParseImageHist('../data/dotmark/picture64_1006.png', N=64)
        #ShowImage(A)
        #ShowImage(B)

        a, b, C = MakeOTproblem(A, B, p=2)

        gamma = 0.1

#        fobj, P = NaiveSinkhorn(a, b, C, eps=gamma, maxit=1000)
#        print(np.sum(P), np.sum(C * P), np.sum(C * P) + gamma * np.sum(P * np.log(P)-P))

        fobj, P = LogSinkhorn(a, b, C, eps=gamma, maxit=1000)
        print(np.sum(P), np.sum(C * P), np.sum(C * P) + gamma * np.sum(P * np.log(P)-P))

        P = ot.sinkhorn(a, b, C, reg=gamma, verbose=True, numItermax=10000, method='sinkhorn_log')
        #ShowImage(P)
        print(np.sum(P), np.sum(C * P), np.sum(C * P) + gamma * np.sum(P * np.log(P)-P))

        P = ot.lp.emd(a, b, C)
        print(np.sum(P), np.sum(C * P))
        #ShowImage(P)

    elif Test == 7:
        A = ParseImageHist('../data/dotmark/picture64_1004.png', N=128)
        B = ParseImageHist('../data/dotmark/picture64_1006.png', N=128)
        #ShowImage(A)
        #ShowImage(B)

        a, b, C = MakeOTproblem(A, B, p=2)

        gamma = 0.1

#        fobj, P = NaiveSinkhorn(a, b, C, eps=gamma, maxit=1000)
#        print(np.sum(P), np.sum(C * P), np.sum(C * P) + gamma * np.sum(P * np.log(P)-P))

        fobj, P = LogSinkhorn(a, b, C, eps=gamma, maxit=1000)
        print(np.sum(P), np.sum(C * P), np.sum(C * P) + gamma * np.sum(P * np.log(P)-P))

        # Study the different option for parameter "method"
        P = ot.sinkhorn(a, b, C, reg=gamma, verbose=True, numItermax=10000, method='sinkhorn_log')
        #ShowImage(P)
        print(np.sum(P), np.sum(C * P), np.sum(C * P) + gamma * np.sum(P * np.log(P)-P))

        P = ot.lp.emd(a, b, C)
        print(np.sum(P), np.sum(C * P))
        #ShowImage(P)
    
    elif Test == 8:
        A = ParseImageHist('../data/dotmark/picture64_1004.png', N=256)
        B = ParseImageHist('../data/dotmark/picture64_1006.png', N=256)
        #ShowImage(A)
        #ShowImage(B)

        a, b, C = MakeOTproblem(A, B, p=2)
        gamma = 0.1

        # DO NOT EVEN TRY THE FOLLOWING:
        # P = ot.lp.emd(a, b, C)
        