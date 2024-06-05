import matplotlib.pyplot as plt

import numpy as np


def ReadJpegBW(filename, xoff=0, yoff=0):
    A = plt.imread(filename)
    A = A[xoff:200+xoff, yoff:200+yoff, 0].astype(float)/255
    return A


def PlotHistogram(A, ax):    
    A = A.flatten()
    I = np.argsort(A)
    ax.hist(A[I], bins=50, color='red', alpha=0.5, density=True)


def MakeHistogram(A):
    I = np.argsort(A)

    (a, bins) = np.histogram(A[I], bins=50)#, color='red', alpha=0.5)
    
    # Normalize to 1
    a = a/sum(a)
    # Return weights and positions
    print('check', len(a), len(bins))
    return a, bins


def PlotThree(A, B, C):
    fig, axs = plt.subplots(2, 3)
    fig.suptitle('Histogram equalizer')

    axs[0,0].imshow(A, cmap='grey')
    PlotHistogram(A, axs[1,0])

    axs[0,1].imshow(B, cmap='grey')
    PlotHistogram(B, axs[1,1])

    axs[0,2].imshow(C, cmap='grey')
    PlotHistogram(C, axs[1,2])

    axs[0,0].axis('off')
    axs[0,1].axis('off')
    axs[0,2].axis('off')

    plt.show()


def PlotTwo(A, B):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Histogram equalizer')

    axs[0,0].imshow(A, cmap='grey')
    PlotHistogram(A, axs[1,0])

    axs[0,1].imshow(B, cmap='grey')
    PlotHistogram(B, axs[1,1])

    axs[0,0].axis('off')
    axs[0,1].axis('off')
    plt.show()


def HistTransfer(A, B, flag=False):
    n, _ = A.shape

    C = A[:].flatten()
    I = np.argsort(C)

    D = B[:].flatten()
    J = np.argsort(D)

    if flag:
        C[I] = D[J]
        return C.reshape(n,n)
    else:
        # Second option
        D[J] = C[I]
        return D.reshape(n,n)


def HistEqual(A):
    n, _ = A.shape

    C = A[:].flatten()
    I = np.argsort(C)

    # To get a uniform histogram:
    #C[I] = np.linspace(0, 1.0, n*n)

    # To get an historgram with a beta distribution
    B = np.random.beta(3, 2, n*n)
    C[I] = B[np.argsort(B)]
    return C.reshape(n,n)
    

A = ReadJpegBW('./foto1.jpeg')
B = ReadJpegBW('./foto3.jpeg', xoff=75, yoff=50)

C = HistTransfer(A, B, True)
PlotThree(A, B, C)

#C = HistEqual(A)
#PlotTwo(A, C)