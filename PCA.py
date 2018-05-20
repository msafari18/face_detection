import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA as mlabPCA
from sklearn.decomposition import PCA as sklearnPCA


class pcaCalculation :

    def __init__(self):
        self.m1 = [10,10]
        self.m2 = [22,10]
        self.cov = [[4, 4], [4, 9]]
        self.X_test_Class = []
        self.average = np.zeros((50, 50))
        self.averageT = np.zeros((50, 50))


    def Q1(self):

        # part one
        class1 = np.random.multivariate_normal(self.m1, self.cov, 1000).T
        class2 = np.random.multivariate_normal(self.m2, self.cov, 1000).T
        plt.plot(class1[0,:], class1[1,:], 'x')
        plt.plot(class2[0,:], class2[1,:], 'x')



        # part two : calculate pca
        samples = np.concatenate((class1, class2), axis=1)

        mlab_pca = mlabPCA(samples.T)
        plt.figure(2)
        plt.plot(mlab_pca.Y[0:1000, 0], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
        plt.plot(mlab_pca.Y[1000:2000, 0], '^', markersize=7, color='yellow', alpha=0.5, label='class2')


        # part three
        plt.figure(1)
        sklearn_pca = sklearnPCA(n_components=1)
        sklearn_transf = sklearn_pca.fit_transform(samples.T)
        p = sklearn_pca.inverse_transform(sklearn_transf)
        plt.figure(1)
        plt.plot(p[0:1000, 0], p[0:1000, 1], 'x')
        plt.plot(p[1000:2000, 0], p[1000:2000, 1], 'x')

        error = ((p - samples.T) ** 2).mean()
        print((error))
        print (np.math.sqrt (error))

        plt.show()

if __name__ == "__main__":
    pc = pcaCalculation()
    pc.Q1()