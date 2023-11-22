# HW 4
import math
import pandas as pd
import copy
import time
import numpy as nump
import matplotlib.pyplot as plt


def file_import(f_name):
    """
    param f_name: for mileage
    :return: import MILEAGE
    """
    # specify features and the response var for file extraction and storage as a table
    feat_names = ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
    f_feat_names = ['x0', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
    df = pd.read_csv(f_name, usecols=feat_names)
    resp_vect = df['mpg'].values.tolist()

    # scale the features
    scale_df = ((df - df.mean()) / df.std())

    # add vector for x0 of 1's; remove the response var
    x_0 = []
    for i in range(len(resp_vect)): x_0.append(1)
    scale_df = scale_df.drop('mpg', axis='columns')
    scale_df.insert(0, "x0", x_0, True)

    # create feature matrix to store all scaled features

    feat_matrix = []
    for feat in f_feat_names:
        feat_matrix.append(scale_df[feat].values.tolist())
    return feat_matrix, resp_vect


def file_import2(filename):
    """
    :param filename: for flowers
    :return: import FLOWER data
    """
    df = pd.read_csv(filename)
    X = df.iloc[:, 1:].copy()
    # X = (X - X.mean()) / X.std()
    X['x0'] = 1
    cols = X.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    X = X[cols]

    X = nump.matrix(X)
    y = nump.transpose(nump.matrix(df['Species']))

    return X, y


def scaling(X):
    """
    :param X: feat matrix
    :return: SCALE TRAIN DATA
    """
    X1 = pd.DataFrame(X)

    X1 = X1.drop(0, axis='columns')
    mean = X1.mean()    # save these for later
    std = X1.std()
    X1 = (X1 - X1.mean()) / X1.std()
    X1.insert(0, 0, 1, True)
    X1 = X1.values.tolist()

    return X1, mean, std


def scaling2(X, mean, std):
    """
    :param X: feat matrix
    :param mean: train data mean
    :param std: train data std
    :return: SCALE TEST DATA
    """
    X = pd.DataFrame(X)
    X = X.drop(0, axis='columns')
    X = (X - mean) / std
    X.insert(0, 0, 1, True)
    X = X.values.tolist()
    return X


def k_feat_calc(X, y, alpha, lamb):
    y_hat = []
    y_hat2 = []
    n = len(X)
    for i in range(n + 1):
        theta, run, cost, it = gradient_descent(X[:i], y, alpha, 'linear', lamb)  # with 1 left out
        y_hat_L = LOOCV(X[:i], y, alpha, lamb)
        y_hat.append(nump.dot(theta, X[:i]))
        y_hat2.append(y_hat_L)
    y_hat.pop(0)    # destroy first set
    y_hat2.pop(0)
    k_calc_errors(y, y_hat, y_hat2, X)


def LOOCV(X, y, alpha, lamb):
    m = len(y)
    n = len(X)
    y_hat = []
    theta = []
    feat_to_predict = []
    for i in range(m):
        # remember original intact matrices for X and y while we remove 1 at a time
        n_X = copy.deepcopy(X)
        n_y = copy.deepcopy(y)
        # begin removals
        feat_to_predict = []
        n_y.pop(i)
        for j in range(n):  # save & remove all features of a specific example to predict later
            feat_to_predict.append(n_X[j][i])
            n_X[j].pop(i)
        theta, run, cost, it = gradient_descent(n_X, n_y, alpha, 'linear', lamb)   # with 1 left out
        y_hat.append(nump.dot(theta, feat_to_predict))
    return y_hat


def k_calc_errors(y, y_hat, y_hat2, X):
    Rs = []
    LOOCV_Rs = []
    for i in range(len(X) - 1):
        error = nump.array(y) - nump.array(y_hat[i]).T
        error_LOOCV = nump.array(y) - nump.array(y_hat2[i]).T
        # calculate R^2 for reg
        SSR = sum(pow(error, 2))
        SST = sum(pow((nump.array(y) - nump.array(nump.mean(y)).T), 2))
        R_Sq = 1 - (SSR / SST)
        # calculate R^2 for LOOCV
        SSR_ = pow(error_LOOCV, 2)
        SST_ = pow((nump.array(y) - nump.array(nump.mean(y)).T), 2)
        LOOCV_R_Sq = 1 - (sum(SSR_) / sum(SST_))
        Rs.append(R_Sq)
        LOOCV_Rs.append(LOOCV_R_Sq)
    plot_rs(Rs, LOOCV_Rs)


def plot_rs(R, LOOCV_R):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.plot(x, R, label='Training R-Squared')
    plt.plot(x, LOOCV_R, label='LOOCV R-Squared')
    plt.legend()
    plt.title("First k-features Training R-Squared and LOOCV R-Squared")
    plt.ylabel("R-Squared Values")
    plt.xlabel("Number of features")
    plt.show()


def gradient_descent(X, y, alpha, model, lamdba):
    # create vector to hold the theta values (past and present), initialize our theta vectors to 0
    start = time.time()
    runtime = 0
    iteration_list = []
    iterations = 0
    threshold = 0.00001
    m = len(y)  # number of observations
    thetas = []
    old_thetas = []
    diff_vect = []
    costs = []
    if model == 'linear':
        for i in range(len(X)):
            thetas.append(0)
            old_thetas.append(0)
            diff_vect.append(0)
    else:
        thetas = nump.zeros(len(X[0]))
        old_thetas = nump.zeros(len(X[0]))
        diff_vect = nump.zeros(len(X[0]))
    not_converged = True
    true_count = 0
    while not_converged:
        # loop over GDA equation algorithm
        for j in range(len(thetas)):    # loop to update each feature
            g_sum = 0
            for i in range(m):  # loop to compute each summation over all observations
                h_theta_x = 0
                if model == 'linear':
                    for n in range(len(thetas)):
                        h_theta_x += (old_thetas[n] * X[n][i])
                elif model == 'cancer':
                    p = -1 * (nump.dot(old_thetas, nump.array(X[i])))
                    h_theta_x = (1 / (1 + math.exp(p)))
                else:
                    p = -1 * (nump.dot(old_thetas, X[i]))
                    h_theta_x = (1 / (1 + math.exp(p)))
                if model == 'linear':
                    g_sum += (X[j][i] * (h_theta_x - y[i]))
                else:
                    g_sum += (X[i][j] * (h_theta_x - y[i]))
            # add back to new theta here
            if lamdba == 0:
                thetas[j] = old_thetas[j] - (alpha / m) * g_sum
            else:   # we are doing regularization
                if j == 0:  # treat j = 0 without reg
                    thetas[j] = old_thetas[j] - (alpha / m) * g_sum
                else:
                    thetas[j] = (old_thetas[j] * (1 - (alpha * lamdba) / m)) - (alpha / m) * g_sum

        # check differences of new thetas will old thetas & check threshold
        for i in range(len(thetas)):
            diff_vect[i] = abs(thetas[i] - old_thetas[i])
            if diff_vect[i] < threshold:
                true_count += 1

        # if all differences are less than threshold, we converged
        if true_count == len(diff_vect):
            not_converged = False
            end = time.time()
            runtime = end-start
        else:
            # update old thetas to continue GDA next iteration
            true_count = 0
            for i in range(len(thetas)):
                old_thetas[i] = thetas[i]
    return thetas, runtime, costs, iteration_list


def plot_distortion_v_run(distortions):
    run = [i for i in range(100)]
    plt.plot(run, distortions)
    plt.title("Distortion vs. Run")
    plt.ylabel("J (Distortion)")
    plt.xlabel("Run")
    plt.show()


def k_means_cluster(k, X, plot_op1):
    rng = nump.random.default_rng(1)    # keep rand generation same
    X = nump.array(X)  # make array for ease
    m = len(X)
    # include loop to run 100 times & calculate distortions
    d_val = 0
    distortions = []
    final_distort_centroid = []
    final_distort_clusters = []
    for run in range(100):
        centroids = []
        converged = False
        for i in range(k):
            rand_int = rng.integers(0, m - 1, 1)
            centroids.append(X[rand_int[0]])
        while not converged:
            clusters = [[] for i in range(k)]   # our cluster groupings
            for j in range(m):
                cmp_dist = []
                for j1 in range(k):
                    # check euclid dist from ex to each centroid
                    e_dist = (nump.linalg.norm(X[j] - centroids[j1])) ** 2
                    cmp_dist.append(e_dist)

                # assign example to a cluster based on closest centroid
                least_dist = min(cmp_dist)
                closest_centroid = cmp_dist.index(least_dist)
                clusters[closest_centroid].append(X[j])
            # make new centroids the avg of examples in that centroid's cluster
            old_centroids = centroids
            # be careful of having an empty cluster
            centroids = [nump.mean(h, axis=0) if h else nump.zeros(len(X[0])) for h in clusters]

            if nump.equal(old_centroids, centroids).all():
                converged = True
        # do a distortion calculation after our algorithm finishes a full clustering
        d_val = distortion(centroids, clusters)
        distortions.append(d_val)
        if min(distortions) == d_val:
            # pick centroids, clusters from run of alg for which distortion is min
            final_distort_centroid = centroids
            final_distort_clusters = clusters
    if plot_op1:
        plot_distortion_v_run(distortions)  # plot distortion vs. run
    return final_distort_clusters, final_distort_centroid, d_val


def distortion(centroids, clusters):
    k = len(centroids)
    c_totals = 0
    for i in range(k):
        # loop over every cluster and sum euclid distances for each ex from its centroid
        e_dist = 0
        m = len(clusters[i])
        for j in range(m):
            e_dist += (nump.linalg.norm(centroids[i] - clusters[i][j])) ** 2
        c_totals += e_dist
    total_m = 0
    for i in range(k):
        total_m += len(clusters[i])
    return c_totals / total_m


def assign_new_k_mean(new_examples, old_centroids):
    k = len(old_centroids)
    new_clust = [[] for i in range(k)]  # cluster groupings
    for i in range(len(new_examples)):
        cmp_dist = []
        for j in range(k):
            # check euclid dist from ex to each centroid
            e_dist = (nump.linalg.norm(new_examples[i] - old_centroids[j])) ** 2
            cmp_dist.append(e_dist)
        # assign example to a cluster based on closest centroid
        least_dist = min(cmp_dist)
        closest_centroid = cmp_dist.index(least_dist)
        new_clust[closest_centroid].append(new_examples[i])
    return new_clust


def confusion_centroids(f_centroids):
    df = pd.DataFrame(f_centroids)
    df = df.drop(0, axis='columns')
    df.columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    df.index = ['Cluster 1', 'Cluster 2', 'Cluster 3']
    print(df)
    return


def k_means_confusion(x, clusters):
    # compare the scaled x's (just like response mat since it's still ordered) to our clusters
    x = nump.array(x)
    new_y = []
    clust_range = len(x) // len(clusters)
    y_hat = []
    # break up each cluster for cross-check
    c_num = 0
    for clus in clusters:
        c_num += 1
        for i in range(len(clus)):
            y_hat.append("Cluster {}".format(c_num))
    # check which examples in the clusters match the response (or ordered x mat)
    for clus in clusters:
        for i in range(len(clus)):
            for j in range(len(x)):
                # search all of x (in 3 flower intervals) for specific ex in a cluster matching an actual flower result
                if nump.array_equal(clus[i], x[j]):
                    if j < clust_range:
                        new_y.append('Setosa')
                    elif clust_range <= j < clust_range*2:
                        new_y.append('Versicolor')
                    elif j >= clust_range*2:
                        new_y.append('Virginica')
    confuse = pd.crosstab(new_y, y_hat, margins=True)
    print(confuse)
    return


def reg_r_plot(X, y, alpha):
    y_hat = []
    y_hat2 = []
    lambs = nump.linspace(0, 10, num=101)   # our set of 101 different lambdas
    for i in range(len(lambs)):     # yes, run 101 times for each lambda
        theta, run, cost, it = gradient_descent(X, y, alpha, 'linear', lambs[i])
        y_hat_L = LOOCV(X, y, alpha, lambs[i])
        y_hat.append(nump.dot(theta, X))
        y_hat2.append(y_hat_L)
    # calculate the R-squared vals
    Rs = []
    LOOCV_Rs = []
    for i in range(101):
        error = nump.array(y) - nump.array(y_hat[i]).T
        error_LOOCV = nump.array(y) - nump.array(y_hat2[i]).T
        # calculate R^2 for reg
        SSR = sum(pow(error, 2))
        SST = sum(pow((nump.array(y) - nump.array(nump.mean(y)).T), 2))
        R_Sq = 1 - (SSR / SST)
        # calculate R^2 for LOOCV
        SSR_ = pow(error_LOOCV, 2)
        SST_ = pow((nump.array(y) - nump.array(nump.mean(y)).T), 2)
        LOOCV_R_Sq = 1 - (sum(SSR_) / sum(SST_))
        Rs.append(R_Sq)
        LOOCV_Rs.append(LOOCV_R_Sq)
    # plot the R-squared values
    plt.plot(lambs, Rs, label='Training R-Squared')
    plt.plot(lambs, LOOCV_Rs, label='LOOCV R-Squared')
    plt.legend()
    plt.title("Training R-Squared vs. LOOCV R-Squared with Regularization")
    plt.ylabel("R-Squared Values")
    plt.xlabel("Lambda (Regularization Parameter)")
    plt.show()


def plot_elbow(data):
    d_list = []
    for i in range(5):
        clust1, centroids1, distort_val = k_means_cluster(i + 1, data, False)
        d_list.append(distort_val)
    k_ = [i + 1 for i in range(5)]
    plt.plot(k_, d_list)
    plt.title("Distortion by (K) Number of Clusters")
    plt.ylabel("J (Distortion)")
    plt.xlabel("K (number of clusters)")
    plt.show()


def pca(x):
    m = len(x)
    # drop the 1's column
    x = pd.DataFrame(x)
    x = x.drop(0, axis='columns')
    x = nump.array(x)
    # step 1: calculate sigma (sample covariance)
    s_cov_mat = (1 / m) * (nump.matmul(nump.transpose(x), x))

    # step 2: get eigenvalues and eigenvectors
    eig_vals, eig_vects = nump.linalg.eig(s_cov_mat)

    # step 3: get first 2 principal components
    U_red = nump.transpose(eig_vects)
    U_red = U_red[:2]

    # step 4: define z feature vector
    z_feat = []
    for i in range(m):
        z_feat.append(nump.matmul(U_red, x[i]))
    z_feat = pd.DataFrame(z_feat)
    # print the scatter plot of data
    p_range = m // 3
    plt.plot(z_feat[0][:p_range], z_feat[1][:p_range], "o", label='Setosa', c='orange')
    plt.plot(z_feat[0][p_range:p_range*2], z_feat[1][p_range:p_range*2], "o", label='Versicolor', c='blue')
    plt.plot(z_feat[0][p_range*2:p_range*3], z_feat[1][p_range*2:p_range*3], "o", label='Virginica', c='green')
    plt.legend()
    plt.ylabel("Z2")
    plt.xlabel("Z1")
    plt.title("PCA Scatter-plot")
    plt.show()

    return z_feat, U_red


if __name__ == '__main__':
    X2, y_f = file_import2('flowers.csv')

    # 1A, separate into test and train
    y_1 = nump.transpose(nump.matrix([1 if val == 'setosa' else 0 for val in y_f]))
    y_2 = nump.transpose(nump.matrix([1 if val == 'versicolor' else 0 for val in y_f]))
    y_3 = nump.transpose(nump.matrix([1 if val == 'virginica' else 0 for val in y_f]))

    rows = [i for i in range(X2.shape[0])]
    trainRows = [i for i in range(0, 40)] + [i for i in range(50, 90)] + [i for i in range(100, 140)]
    testRows = sorted(list(set.difference(set(rows), set(trainRows))))

    XTrain = X2[trainRows, :]  # scale me
    yTrain = y_f[trainRows, :]

    y1Train = y_1[trainRows, :]
    y2Train = y_2[trainRows, :]
    y3Train = y_3[trainRows, :]

    XTest = X2[testRows, :]  # scale me with XTrain's mean and std
    yTest = y_f[testRows, :]

    y1Test = y_1[testRows, :]
    y2Test = y_2[testRows, :]
    y3Test = y_3[testRows, :]

    # 1B - Scaling
    XTrainScaled, X_mean, X_std = scaling(XTrain)
    XTestScaled = scaling2(XTest, X_mean, X_std)
    print("----------------------------")

    # 1C - k means (k = 3)
    """COMPARE WITH DATA FROM HW2 3D"""
    clust, centroids, distort_v = k_means_cluster(3, XTrainScaled, True)
    print("\nThe resulting k-means centroids:\n")
    confusion_centroids(centroids)
    print("\nTraining data: Response vs. Cluster Assignment:\n")
    k_means_confusion(XTrainScaled, clust)

    # 1D
    """COMPARE WITH DATA FROM HW2 3E"""
    # call function for assigning new examples (test data) into clusters using final centroids
    # then call k_means_confusion again with new testing clusters and XTestScaled
    print("\nTesting data: Response vs. Cluster Assignment:\n")
    test_clusters = assign_new_k_mean(XTestScaled, centroids)
    k_means_confusion(XTestScaled, test_clusters)

    """2"""
    # 2, Plot elbow graph
    plot_elbow(XTrainScaled)

    """3"""
    # 3A-3B, PCA with training data
    zTrain, U = pca(XTrainScaled)
    print("\nThe first two principal components:\n")
    U1 = pd.DataFrame(nump.transpose(U))
    U1.columns = ['U1', 'U2']
    print(U1)
    print("\nThe new Z feature matrix:\n")
    zTrain.columns = ['Z1', 'Z2']
    print(zTrain)

    # 3C, do same with test data
    zTest, U_ = pca(XTestScaled)

    """4"""
    # 4A, k-means for PCA training data COMPARE TO 1C
    print("\nTraining data: Response vs. Cluster Assignment:\n")
    z_clust, z_centroids, z_distort_v = k_means_cluster(3, zTrain, False)
    k_means_confusion(zTrain, z_clust)

    # 4B, k-means for PCA testing data, COMPARE TO 1D
    print("\nTesting data: Response vs. Cluster Assignment:\n")
    z_clust1, z_centroids1, z_distort_v1 = k_means_cluster(3, zTest, False)
    k_means_confusion(zTest, z_clust1)

    """5"""
    # 5A, Graph Training R^2 vs LOOCV R^2 for first k-mean features
    X_, y_ = file_import('mileage.csv')
    a = 0.2
    k_feat_calc(X_, y_, a, 0)

    # 5B, Interpret plot

    # 5C, Graph Training R-Squared vs. LOOCV R-Squared with Regularization
    reg_r_plot(X_, y_, a)