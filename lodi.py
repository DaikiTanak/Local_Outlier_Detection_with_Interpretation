import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd

import math
import copy
from tqdm import tqdm_notebook as tqdm
import pyper
from statistics import mean, median,variance,stdev
from scipy import linalg

import pandas as pd

import time

class LODI(object):
    """
            LODI is the unsupervised anomaly detection method based on subspace search
        that separate inliers and outliers most,
        and the interpretation method of detected anomalies based on the subspace weights.

        Args :
            Dataset which may include outliers : X (num of data * dimension)
            lower_bound of neighbors : min_k
            initial neighbors of kNN. : max_k
            How many anomalys to detect : K
            The proportion of the interpretation usage : lambda

        Return :
            1.Anomaly candidates set.
            2.The effective feature set why anomaly is exceptional.
            3.The weights of effective features.

        Input X can be a high dimensional matrix.
    """

    def __init__(self, X, min_k=20, max_k=40):
        self.X = X
        self.min_k = min_k
        self.max_k = max_k

    def search_neighbors(self):
        """
            Search neighbors based by minimizing Renyi-entropy
        caluculated from neighbors.
        """

        NN = NearestNeighbors(n_neighbors=self.max_k+1, metric="minkowski", p=2)
        NN.fit(self.X)

        distance, index_list_list = NN.kneighbors(self.X, n_neighbors=self.max_k+1, return_distance=True)

        neighbors_list = []

        for index_list in index_list_list:
            knn_neighbors = []
            if_first=True
            for index in index_list:
                if if_first:
                    if_first = False
                    continue
                else:
                    knn_neighbors.append(self.X[index])
            neighbors_list.append(knn_neighbors)

        QE_list = []

        #Kernel function
        def gaussian(a, b, band_width):
            norm = np.sqrt(np.dot(a-b, a-b))
            D = len(a)
            return (1 / np.sqrt(2*np.pi*band_width)**(D/2)) * np.exp(-norm**2 / (2*(band_width**2)))

        def Quadratic_Entoropy(neighbors):
            QE = 0
            for x_i in neighbors:
                for x_j in neighbors:
                    QE += gaussian(x_i, x_j, 1)
            if QE == 0:
                QE += 1e-10
            QE = - math.log(QE/len(neighbors)**2)
            return QE

        true_neighbors_list, true_neighbors_index_list = [], []

        for neighbors, target in zip(tqdm(neighbors_list), self.X):
            QE = Quadratic_Entoropy(neighbors)
            # select optimal subset.
            potential_list = []

            for i in range(len(neighbors)):
                neighbors_copy = copy.deepcopy(neighbors)
                del(neighbors_copy[i])
                new_neighbors = neighbors_copy
                new_QE = Quadratic_Entoropy(new_neighbors)
                potential_list.append(QE - new_QE)


            #上から順にentoropyに貢献した要素
            sorted_potential_index = np.argsort(potential_list)[-1::-1]
            average_potential = np.mean(potential_list)

            remove_index = []
            for index in sorted_potential_index:
                if potential_list[index] > average_potential:
                    if len(remove_index) >= (self.max_k-self.min_k):
                        break
                    remove_index.append(index)

                else:
                    break
            true_neighbor_index = list(set(sorted_potential_index) - set(remove_index))
            true_neighbors = [neighbors[index] for index in true_neighbor_index]

            true_neighbors_list.append(np.array(true_neighbors))
            true_neighbors_index_list.append(true_neighbor_index)

            #QE = Quadratic_Entoropy(true_neighbors)
            #QE_list.append(QE)

        self.neighbors_list = true_neighbors_list
        self.neighbors_index_list = true_neighbors_index_list
        return np.array(true_neighbors_list)


    def anomalous_degree(self, svd="normal"):
        """
            Compute Local anomalous degree of every data.
            When getting optimal w, I use [Tan+ 2018] method.
            "Sparse Generalized Eigenvalue Problem: Optimal Statistical Rates via Truncated Rayleigh Flow"
        """



        X = self.X
        neighbors_list = self.neighbors_list

        w_list = []
        lambda_list = []
        AD_list = []


        avg_time = 0
        for target, neighbors in zip(tqdm(X), neighbors_list):

            #print(np.matrix(neighbors).shape)
            e = np.matrix(np.ones(len(neighbors))).T
            R = np.matrix(neighbors).T
            A = R - (R.dot(e).dot(e.T)/len(neighbors))
            B = np.matrix([target - x for x in neighbors]).T

            # J(w)AA.T = BB.Tw
            # These metrices are d*d.
            AA_t = A.dot(A.T)
            BB_t = B.dot(B.T)


            
            if svd == "normal":
                U, s, V = linalg.svd(A)

                singular_values_sum = np.sum(s)
                rank_threshold = 0.95 * singular_values_sum

                memo = 0
                rank = 0
                for singular_value in s:
                     memo += singular_value
                     rank += 1
                     if memo > rank_threshold:
                         break

                Ur = U[:, :rank]
                sr = np.matrix(linalg.diagsvd(s[:rank], rank,rank))
                Vr = V[:rank, :]
                # Aの近似
                A_ = np.asarray(Ur*sr*Vr)
                #print(A, A_)

                sr_ = sr**(-2)
                new_matrix = Ur*sr_*Ur.T*B*B.T
                lambda_, vr = linalg.eig(new_matrix, right=True, left=False)
                #print(lambda_)

                w = vr[:,np.argmax(lambda_)]

            
            now = time.time()
            # Randomized SVD
            if svd == "random":

                U, s, V = randomized_svd(A, 10, random_state=46)

                singular_values_sum = np.sum(s)
                rank_threshold = 1.0 * singular_values_sum

                memo = 0
                rank = 0
                for singular_value in s:
                     memo += singular_value
                     rank += 1
                     if memo > rank_threshold:
                         break

                Ur = U[:, :rank]
                sr = np.matrix(linalg.diagsvd(s[:rank], rank,rank))
                Vr = V[:rank, :]
                # Aの近似
                A_ = np.asarray(Ur*sr*Vr)

                sr_ = sr**(-2)
                new_matrix = Ur*sr_*Ur.T*B*B.T
                lambda_, vr = linalg.eig(new_matrix, right=True, left=False)
                w = vr[:,np.argmax(lambda_)]



            
            if svd == "sparse":
                r = pyper.R()

                n=50
                p=25

                # Using https://arxiv.org/abs/1604.08697 to get optimal w.
                # The R implementation is here: https://cran.r-project.org/web/packages/rifle/index.html
                # set parameters.
                r("source(file='function.R')")
                r("source(file='initial.convex.R')")
                r("source(file='rifle.R')")
                r.assign("A", BB_t)
                r.assign("B", AA_t)
                r.assign("K", 1)
                r.assign("lambda", 2*np.sqrt(np.log(p)/n))
                r("v = initial.convex(A,B,lambda,K,nu=1,trace=FALSE)")
                r("init = eigen(v$Pi+t(v$Pi))$vectors[,1]")
                r("k <- 10")
                r("xprime <- rifle(A,B,init,k,0.01,1e-3)")

                w = np.matrix(r.get("xprime"))
                wAAw = w.T.dot(AA_t).dot(w)
                wBBw = w.T.dot(BB_t).dot(w)
                max_lambda = np.array(wBBw/wAAw)[0][0]
                lambda_list.append(max_lambda)
            
            avg_time += time.time()-now

            w_list.append(w)
            # Compute Anomlous Degree.
            # Variance of neighbors on the projection.
            projected_neighbors = np.matrix(neighbors).dot(w)
            var_neighbors = np.var(np.array(projected_neighbors))
            mean_neighbors = np.mean(np.array(projected_neighbors))

            wo = np.array(w.T.dot(target))#[0][0]
            AD = np.sqrt(((wo - mean_neighbors)/var_neighbors)**2)
            AD = max(AD, np.sqrt(var_neighbors))
            AD_list.append(AD)

        print("avg svd time : {}".format(avg_time/len(X)))
        #Compute Local Anomolous Degree
        LAD_list = []
        for target_index, neighbors_index in zip(range(len(X)), self.neighbors_index_list):
            #print("target and its neighbors:",target_index, neighbors_index)
            target_AD = AD_list[target_index]
            neighbors_AD = []
            for nei_index in neighbors_index:
                neighbors_AD.append(AD_list[nei_index])
            LAD_list.append(target_AD / np.mean(neighbors_AD))

        self.LAD_list = LAD_list
        self.w_list = np.array(w_list)


        #outlier_num=2

        #outlier_index = np.argsort(LAD_list)[-1::-1][:outlier_num]
        #for index in outlier_index:
        #    print("outlier:", X[index])

        return LAD_list, w_list

    def interpret_outliers(self, lambda_=0.8, outlier_num=10):
        """
            Give interpretation to data points.
        """
        LAD_list = self.LAD_list
        w_list = self.w_list

        interpretation = []
        importances_list = []
        for w in w_list:
            w = w.flatten()
            abs_w = abs(w)
            w_sum = np.sum(abs_w)
            threshold = lambda_ * w_sum
            w_sorted, w_sorted_index = np.sort(abs_w)[-1::-1], np.argsort(abs_w)[-1::-1]
            #print(w_sorted_index)
            weight_sum = 0
            effective_feature_index = []
            importances = []
            for element, index in zip(w_sorted, w_sorted_index):
                if weight_sum > threshold:
                    break
                else:
                    weight_sum += abs(element)
                    effective_feature_index.append(index)
                    importances.append(abs(element)/w_sum)
            interpretation.append(effective_feature_index)
            importances_list.append(importances)

        #outlier_num = 4
        outliers_index = np.argsort(LAD_list)[-1::-1][:outlier_num]
        print("top{} outliers".format(outlier_num))
        for index in outliers_index:
            pass
            #print("data index:", index)
            #print("data:", self.X[index])
            #print("効いてる次元:", interpretation[index])
            #print("importances:", importances_list[index])
            #print("-----------------------------------------------------------")

        return outliers_index, interpretation, importances_list

    def run(self, lam):
        _ = self.search_neighbors()
        LAD_list, w_list = self.anomalous_degree()
        explanation = self.interpret_outliers(lambda_=lam)

        return LAD_list, explanation


if __name__ == "__main__":
    import pickle

    train = pd.read_csv("input.csv")
    columns = train.columns

    X = np.array(train.iloc[:,1:])
    y = np.array(train["class"])

    
    normal, sick = [], []
    for data,label in zip(X,y):
        if label == -1:
            normal.append(data)
        else:
            sick.append(data)
            
    X = np.vstack((normal, [sick[0]]))


    lodi = LODI(X, min_k=10, max_k=20)
    LAD_, exp_ = lodi.run(0.8)

    with open('LAD.pickle', mode='wb') as f:
        pickle.dump(LAD_, f)
    with open('exp.pickle', mode='wb') as f:
        pickle.dump(exp_, f)







