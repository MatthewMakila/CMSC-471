# HW 2
import math

import pandas as pd
import copy
import time
import numpy as nump
import matplotlib.pyplot as plt


def file_import(f_name):    # for mileage data
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


def file_import2(f_name):   # for cancer data
    df = pd.read_csv(f_name)
    resp_vect = df['LungCancer'].values.tolist()
    # add vector for x0 of 1's; remove the response var
    x_0 = []
    for i in range(len(resp_vect)): x_0.append(1)
    df = df.drop('LungCancer', axis='columns')
    df.insert(0, "x0", x_0, True)
    feat_names = ['x0', 'Smoking']
    feat_mat = df.values.tolist()

    return feat_mat, resp_vect


def file_import3(filename):
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


def mpg_estimate(X, y, Theta):
    # create a vector of estimations (y hat)
    y_hat = []
    theta = nump.array(Theta)
    for i in range(len(y)):
        sample = []
        for j in range(len(X)):
            sample.append(X[j][i])
        # calculate a y hat with numpy: multiply theta transpose by that sample's features
        new_X = nump.array(sample)
        y_hat.append(nump.dot(theta, new_X))
    # calculate the error between observed and predicted
    calc_errors(y, y_hat, X)


def calc_errors(y, y_hat, X):
    error = nump.array(y) - nump.array(y_hat).T
    data = {'Observed mileage': y, 'Estimated mileage': y_hat, 'Error': error}
    df = pd.DataFrame(data=data)
    print(df)
    # calculate R^2 and adjusted R^2
    SSR = pow(error, 2)
    SST = pow((nump.array(y) - nump.array(nump.mean(y)).T), 2)
    R_Sq = 1 - (sum(SSR) / sum(SST))
    A_R_Sq = 1 - (1 - R_Sq) * ((len(y) - 1) / (len(y) - len(X)))
    print("R-Squared: ", R_Sq)
    print("Adjusted R-Squared: ", A_R_Sq)


def LOOCV(X, y, alpha, model):
    m = len(y)
    n = len(X)
    y_hat = []
    if model == 'linear':
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
                #print(feat_to_predict)
            # print(feat_to_predict)
            theta, run, cost, it = gradient_descent(n_X, n_y, alpha, 'linear')   # with 1 left out
            y_hat.append(nump.dot(theta, feat_to_predict))
        calc_errors(y, y_hat, X)
    else:   # logistic regression model
        print()


def k_feat_calc(X, y, alpha):
    y_hat = []
    n = len(X)
    for i in range(n + 1):
        theta, run, cost, it = gradient_descent(X[:i + 1], y, alpha, 'linear')  # with 1 left out
        y_hat.append(nump.dot(theta, X[:i + 1]))
    y_hat.pop(0)    # destroy first set
    k_calc_errors(y, y_hat, X)


def k_calc_errors(y, y_hat, X):
    Rs = []
    A_Rs = []
    for i in range(len(X) - 1):
        error = nump.array(y) - nump.array(y_hat[i]).T
        # calculate R^2 and adjusted R^2
        SSR = sum(pow(error, 2))
        SST = sum(pow((nump.array(y) - nump.array(nump.mean(y)).T), 2))
        R_Sq = 1 - (SSR / SST)
        A_R_Sq = 1 - (1 - R_Sq) * ((len(y) - 1) / (len(y) - len(X)))
        Rs.append(R_Sq)
        A_Rs.append(A_R_Sq)
    plot_rs(Rs, A_Rs)


def gradient_descent(X, y, alpha, model):
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
            thetas[j] = old_thetas[j] - (alpha / m) * g_sum
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


def pred_cancer(X, y, theta):
    y_hat = []
    threshold = 0.5
    m = len(y)
    for i in range(m):
        p = -1 * (nump.dot(theta, nump.array(X[i])))
        h_theta_x = (1 / (1 + math.exp(p)))
        if h_theta_x <= threshold:  # if predictor is less than 0.5, we have class 0
            y_hat.append(0)
        else:
            y_hat.append(1)
    TP = 0
    TN = 0
    P = 0
    N = 0
    PP = 0

    for i in range(m):
        # count our classifications
        if y_hat[i] == y[i]:    # correct if match
            if y_hat[i] == 1:   # TP if eq 1, else TN
                TP += 1
            else:
                TN += 1
        if y[i] == 1:
            P += 1
        else:
            N += 1
        if y_hat[i] == 1:
            PP += 1

    confuse = pd.crosstab(y, y_hat, rownames=['Actual'], colnames=
                          ['Predicted'], margins=True)
    correct_class = (TP + TN) / m
    precision = TP / PP
    recall = TP / P
    specificity = TN / N
    f1 = 2 / (pow(precision, -1) + pow(recall, -1))

    print(confuse)
    print("Correct % of classifications:\t", correct_class)
    print("Precision:\t", precision)
    print("Recall:\t", recall)
    print("Specificity:\t", specificity)
    print("F1:\t", f1)
    print("Odds Ratio:\t", math.exp(theta[1]))


def plot_rs(R, A_R):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.plot(x, R, label='R-Squared')
    plt.plot(x, A_R, label='Adjusted R-Squared')
    plt.legend()
    plt.ylabel("R-Squared Values")
    plt.xlabel("Number of features")
    plt.show()


def scaling(X):
    X1 = pd.DataFrame(X)

    X1 = X1.drop(0, axis='columns')
    mean = X1.mean()    # save these for later
    std = X1.std()
    X1 = (X1 - X1.mean()) / X1.std()
    X1.insert(0, 0, 1, True)
    X1 = X1.values.tolist()

    return X1, mean, std


def scaling2(X, mean, std):
    X = pd.DataFrame(X)
    X = X.drop(0, axis='columns')
    X = (X - mean) / std
    X.insert(0, 0, 1, True)
    X = X.values.tolist()
    return X


def pred_flower(X, y, theta):
    y = y.tolist()
    y = [j for sub in y for j in sub]
    y_hat = []
    m = len(y)

    for i in range(m):
        h_set = (1 / (1 + math.exp(-1 * (nump.dot(theta[0], X[i])))))
        h_ver = (1 / (1 + math.exp(-1 * (nump.dot(theta[1], X[i])))))
        h_vir = (1 / (1 + math.exp(-1 * (nump.dot(theta[2], X[i])))))
        p = max(h_set, h_ver, h_vir)
        if p == h_set:
            y_hat.append('setosa')
        elif p == h_ver:
            y_hat.append('versicolor')
        else:
            y_hat.append('virginica')

    T = 0
    Set_T = 0
    Ver_T = 0
    Vir_T = 0

    for i in range(m):
        # count our classifications
        if y_hat[i] == y[i]:  # correct if match
            T += 1
            if y_hat[i] == 'setosa':
                Set_T += 1
            if y_hat[i] == 'versicolor':
                Ver_T += 1
            if y_hat[i] == 'virginica':
                Vir_T += 1

    confuse = pd.crosstab(y, y_hat, rownames=['Actual'], colnames=['Predicted'], margins=True)
    correct_class = T / m
    Set_correct = Set_T / (m/3)
    Ver_correct = Ver_T / (m/3)
    Vir_correct = Vir_T / (m/3)

    print(confuse)
    print("Overall Correct % of Classifications:\t", correct_class)
    print("Setosa Correct % of Classifications:\t", Set_correct)
    print("Versicolor Correct % of Classifications:\t", Ver_correct)
    print("Virginica Correct % of Classifications:\t", Vir_correct)


if __name__ == '__main__':
    X_, y_ = file_import('mileage.csv')
    a = 0.2
    """1"""
    # 1A
    #Thetas, runtime, costs, iterations = gradient_descent(X_, y_, a, 'linear')
    #mpg_estimate(X_, y_, Thetas)

    # 1B
    #LOOCV(X_, y_, a, 'linear')

    # 1C
    #k_feat_calc(X_, y_, a)

    """2"""
    # 2A
    X1, y1 = file_import2('cancer.csv')
    Thetas2, runtime2, costs2, iterations2 = gradient_descent(X1, y1, a, 'cancer')

    # 2B
    #pred_cancer(X1, y1, Thetas2)

    # 2C
    # From the calculated Odds ratio, I think it can be determined that smokers are about 1.87 times as likely to
    # develop cancer than non-smokers are.

    """3"""
    # 3A - Divide into test and train data sets
    X2, y_f = file_import3('flowers.csv')
    #print(X2)
    #print(y_f)
    y_1 = nump.transpose(nump.matrix([1 if val == 'setosa' else 0 for val in y_f]))
    y_2 = nump.transpose(nump.matrix([1 if val == 'versicolor' else 0 for val in y_f]))
    y_3 = nump.transpose(nump.matrix([1 if val == 'virginica' else 0 for val in y_f]))

    rows = [i for i in range(X2.shape[0])]
    trainRows = [i for i in range(0, 40)] + [i for i in range(50, 90)] + [i for i in range(100, 140)]
    testRows = sorted(list(set.difference(set(rows), set(trainRows))))

    XTrain = X2[trainRows, :]   # scale me
    yTrain = y_f[trainRows, :]

    y1Train = y_1[trainRows, :]
    y2Train = y_2[trainRows, :]
    y3Train = y_3[trainRows, :]

    XTest = X2[testRows, :]     # scale me with XTrain's mean and std
    yTest = y_f[testRows, :]

    y1Test = y_1[testRows, :]
    y2Test = y_2[testRows, :]
    y3Test = y_3[testRows, :]

    # 3B - Scaling
    XTrainScaled, X_mean, X_std = scaling(XTrain)
    XTestScaled = scaling2(XTest, X_mean, X_std)

    # 3C - Learn probabilities with 3 separate models for classification
    # model 1 (setosa)
    # Thetas_s, runtime_s, costs_s, iterations_s = gradient_descent(XTrainScaled, y1Train, a, 'logistic')
    # print(Thetas_s)
    T1 = [-5.24334614, -1.75999339,  3.9898382,  -4.42729052, -4.45791134]
    T1 = nump.array(T1)
    # model 2 (versicolor)
    # Thetas_versi, runtime_versi, costs_versi, iterations_versi = gradient_descent(XTrainScaled, y2Train, a, 'logistic')
    # print(Thetas_versi)
    T2 = [-0.94861559,  0.27162912, -1.38605387,  0.61796515, -0.89460269]
    T2 = nump.array(T2)
    # model 3 (virginica)
    # Thetas_virg, runtime_virg, costs_virg, iterations_virg = gradient_descent(XTrainScaled, y3Train, a, 'logistic')
    # print(Thetas_virg)
    T3 = [-18.70584143,  -1.99562271,  -2.78318499,  15.79802893,  12.91789495]
    T3 = nump.array(T3)

    # 3D - Predictions using training data
    pred_flower(XTrainScaled, yTrain, [T1, T2, T3])

    # 3E - Predictions using test data
    pred_flower(XTestScaled, yTest, [T1, T2, T3])

