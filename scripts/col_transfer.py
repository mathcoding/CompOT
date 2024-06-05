import numpy as np
import matplotlib.pyplot as plt
from random import sample, seed
seed(13)
np.random.seed(13)

# Import the POT library
import ot


def LoadImage(filename):
    A = plt.imread(filename).astype(np.float64)/255
    return A

def ShowImage(A):
    fig, ax = plt.subplots()
    plt.imshow(A)
    ax.autoscale()
    ax.set_aspect('equal', 'box')
    plt.axis('off')
    plt.show()

def DisplayCloud(A, samples=100):
    n, m, l = A.shape
    print('shape', n, m, l)
    C = A.reshape(n*m, 3)
    s = sample(range(0, m*n), samples)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.scatter(x=C[s, 0], y=C[s, 1], zs=C[s, 2], s=100, c=C[s])
    plt.show()

def PointSamples(A, samples=100):
    n, m, l = A.shape
    C = A.reshape(n*m, 3)
    s = sample(range(0, m*n), samples)
    return C[s]

def ColorMap(H1, H2):
    n, _ = H1.shape
    m, _ = H2.shape

    C = np.zeros((n,m))

    A = np.ones(n)
    B = np.ones(m)
    # Build cost matrix
    for i in range(n):
        for j in range(m):
            C[i,j] = np.linalg.norm(H1[i]-H2[j])
            
    pi = ot.emd(A, B, C)

    # To use the sinkhorn algorithm (but you should select a "reg"-ularization value)
    #pi = ot.sinkhorn(A, B, C, reg=0.01, verbose=True)

    # Optimal solution found
    # Map node i to j if pi[i,j] == 1.0
    M = []
    for i in range(len(H1)):
        T = []
        for j in range(len(H2)):
            if pi[i,j] > 0.01:
                T.append( (j, pi[i,j]) )
        M.append(T)
    return M


# Distance of cartesian product between vectors in A and B
from scipy.spatial.distance import cdist
def ClosestRGB(A, B):
    return np.argmin(cdist(A, B), axis=1)

def Wheel(Ps, d):
    if len(Ps) == 0:
        return d
    if len(Ps) == 1:
        return Ps[0][0]
    # If more than a single element, return with probabilities
    As = [p[0] for p in Ps]
    Pr = [p[1] for p in Ps]
    Pr[0] = Pr[0] + 1.0-sum(Pr)
    Ws = np.random.choice(As, size=1, p=Pr)
    return Ws[0]


B = LoadImage('./ferrari.jpg')
A = LoadImage('./margherita.jpeg')

H1 = PointSamples(A, 500)
H2 = PointSamples(B, 500)

#ShowImage(A)
#ShowImage(B)

CMAP = ColorMap(H1, H2)


n, m, _ = A.shape
C = A.reshape(n*m, 3)

Y = ClosestRGB(C, H1)
H4 = np.array([H2[Wheel(CMAP[i], i)] for i in Y])
H5 = H4.reshape(n, m, 3)

ShowImage(H5)

# NOTE:
# For an alternative demo on optimal color transfer, see the following link
# from the Python Optimal Transport library:
# https://pythonot.github.io/auto_examples/domain-adaptation/plot_otda_mapping_colors_images.html