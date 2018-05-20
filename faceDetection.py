from PIL import Image

#...
from np.magic import np
from numpy.ma import array
from scipy.misc import toimage
from sklearn import linear_model
import matplotlib.pyplot as plt
from numpy import linalg as LA


class ImageRecognation :

    def __init__(self):
        self.X = []
        self.XClass = []
        self.X_test = []
        self.X_test_Class = []
        self.average = np.zeros((50, 50))
        self.averageT = np.zeros((50, 50))

    def imageProcessing(self) :
        with open('faces\\train.txt') as f:
            f1 = f.readlines()

        # part A , B
        for line in f1:

            path = line.split(' ')
            arr = array(Image.open(path[0]))
            self.X.append(arr)
            self.average += arr
            path[1].replace("\n" , "")
            self.XClass.append(path[1])

        # toimage(self.X[2]).show()

        # part C
        self.average = self.average / 540
        # toimage(self.average).show()

        for i in range(len(self.X)):
            self.X[i] = self.X[i] - self.average
        # toimage(self.X[2]).show()


        # part E
        # svd :D

        x = np.reshape(self.X, (540, 2500))
        U, s, V = np.linalg.svd(x, full_matrices=False)

        # for i in range(0,10):
        newV = np.reshape(V[10], (50,50))
        # arr = array(V[i])
        # toimage(newV).show()


        # part F

        error1 = []
        for r in range (1,200) :
            error = 0
            sigma = np.zeros((540, 540))
            for i in range(r):
                sigma[i][i] = s[i]
            tmp = np.matmul(U, sigma)
            tmp2 = np.matmul(tmp, V)
            error1.append(LA.norm(tmp2 - x, 'fro'))
            # xr = np.reshape(tmp2[1], (50, 50))
            #
            #
            # t =  np.matrix(U[i][:,:r]) * np.diag(s[i][:r]) *  np.matrix(V[i][:r,:])
            # error += (np.abs(t - self.X[i])).sum() / (len(t) * len(t))
            # error1.append(error / 540)

        plt.plot(error1)
        plt.show()



        # part G

        error2 = []
        self.readTestImage();
        for r in range(1,200) :

            # for train set
            F = np.zeros((540, r))
            newX = np.reshape(self.X, (540, 2500))
            newV = np.reshape(V[0:r].T, (2500, r))
            np.matmul(newX, newV, F)
            if r == 10 :
                print ('F')
                print (F)

            # for test set
            F_test = np.zeros((100, r))
            newX_test = np.reshape(self.X_test, (100, 2500))
            np.matmul(newX_test, newV, F_test)
            if r == 10 :
                print ("F_test")
                print (F_test)

            # part H

            # implement a logistic regression to predict test set
            e = 0
            logreg = linear_model.LogisticRegression(C=1e5)
            logreg.fit(F, self.XClass)
            predict = logreg.predict(F_test)
            for j in range(len(self.X_test_Class)):
                if self.X_test_Class[j] != predict[j]:
                    e += 1
            #         number of wrong predict

            error2.append(e)
        print (e)
        plt.plot(error2)
        plt.show()


    def readTestImage(self):

        with open('faces\\test.txt') as f:
            f2 = f.readlines()

        for line in f2:
            path = line.split(' ')
            arr = array(Image.open(path[0]))
            self.X_test.append(arr)
            # self.averageT += arr
            path[1].replace("\n", "")
            self.X_test_Class.append(path[1])

        # self.averageT = self.averageT / 100
        for i in range(len(self.X_test)):
            self.X_test[i] = self.X_test[i] - self.average

if __name__ == "__main__":

    iR = ImageRecognation()
    iR.imageProcessing()
