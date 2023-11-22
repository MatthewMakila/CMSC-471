# HW 3
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
    # calculate R^2 and adjusted R^2
    SSR = pow(error, 2)
    SST = pow((nump.array(y) - nump.array(nump.mean(y)).T), 2)
    R_Sq = 1 - (sum(SSR) / sum(SST))
    A_R_Sq = 1 - (1 - R_Sq) * ((len(y) - 1) / (len(y) - len(X)))
    print("R-Squared: ", R_Sq)
    print("Adjusted R-Squared: ", A_R_Sq)


def LOOCV(X, y, alpha, lamb, alg):
    m = len(y)
    n = len(X)
    y_hat = []
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
        if alg == 'gda':
            theta, run, cost, it = gradient_descent(n_X, n_y, alpha, 'linear', lamb)   # with 1 left out
            y_hat.append(nump.dot(theta, feat_to_predict))
        elif alg == 'norm':
            theta = normal_equations(n_X, n_y, lamb)
            y_hat.append(nump.dot(theta, feat_to_predict))
    calc_errors(y, y_hat, X)


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


def normal_equations(feat, resp, lamb):
    if lamb == 0:
        X1 = nump.array(feat).T
        y1 = nump.array(resp)
        temp = nump.dot(X1.T, X1)
        temp = nump.linalg.inv(temp)
        temp = nump.dot(temp, X1.T)
        temp = nump.dot(temp, y1)
        Theta = temp
    else:   # regularize
        X1 = nump.array(feat).T
        y1 = nump.array(resp)
        temp = nump.dot(X1.T, X1)
        # add new part here:    + (lamb * I)
        n = len(feat)
        row = nump.zeros(n)
        mat_I = nump.identity(n)
        m_df = pd.DataFrame(mat_I)
        m_df.loc[0] = row
        mat_I = m_df.to_numpy()
        mat_I = nump.dot(lamb, mat_I)
        temp = temp + mat_I
        temp = nump.linalg.inv(temp)
        temp = nump.dot(temp, X1.T)
        temp = nump.dot(temp, y1)
        Theta = temp
    return Theta


def ridge_trace_plot(X, y, alpha, alg):
    my_thetas = []
    lambs = nump.linspace(0, 10, num=101)   # our set of 101 different lambdas
    print(lambs)
    for i in range(len(lambs)):     # yes, we run it 101 times for each lambda
        if alg == 'gda':
            theta, run, cost, it = gradient_descent(X, y, alpha, 'linear', lambs[i])
            my_thetas.append(theta)
        elif alg == 'norm':
            theta = normal_equations(X, y, lambs[i])
            my_thetas.append(theta)
    my_thetas = pd.DataFrame(my_thetas)
    my_thetas = my_thetas.transpose().values.tolist()
    # no need to plot theta 0
    for i in range(len(my_thetas) - 1):
        plt.plot(lambs, my_thetas[i + 1], label="theta {}".format(i + 1))

    plt.legend()
    plt.title("Ridge Trace")
    plt.ylabel("Standardized Coefficient")
    plt.xlabel("Ridge Parameter")
    plt.show()


if __name__ == '__main__':
    X_, y_ = file_import('mileage.csv')
    a = 0.2
    """1"""

    # 1A: calculate regular MLR and then LOOCV MLR
    print("\n-----------------1A---------------\n")
    Thetas, runtime, costs, iterations = gradient_descent(X_, y_, a, 'linear', 0)
    print("Training:")
    mpg_estimate(X_, y_, Thetas)
    print("LOOCV:")
    LOOCV(X_, y_, a, 0, 'gda')

    df = pd.DataFrame()
    df.insert(0, "Model Parameters (No Regularization)", Thetas, True)
    print(df)

    # 1B: regular MLR and LOOCV MLRs with Regularization
    print("\n-----------------1B---------------\n")
    reg_param = 6.5
    Thetas2, runtime2, costs2, iterations2 = gradient_descent(X_, y_, a, 'linear', reg_param)
    print("Training:")
    mpg_estimate(X_, y_, Thetas2)
    print("LOOCV:")
    LOOCV(X_, y_, a, reg_param, 'gda')
    df2 = pd.DataFrame()
    df2.insert(0, "Model Parameters (Regularization)", Thetas2, True)
    print(df2)

    # 1C: Write about how results from 1A and 1B compare
    # We can see there is a significant improvement in the R-squared values from the initial
    # models to the ones trained using regularization. The models that used regularization have a much better
    # LOOCV fit, with the training fit decreasing, as expected. We can also notice the shrinkage in the model
    # parameters, as expected by the regularization's change to the cost function, since we were multiplying theta by
    # a number smaller than 1 while training.

    # 1D: Ridge Trace Plot
    ridge_trace_plot(X_, y_, a, 'gda')

    # 1E: Write about how to choose Regularization parameter based upon ridge plot results

    """2"""

    # Repeat 1A-1E but for Normal Equations Method instead of GDA
    # 2A
    print("\n-----------------2A---------------\n")
    norm_theta = normal_equations(X_, y_, 0)
    print("Training:")
    mpg_estimate(X_, y_, norm_theta)
    print("LOOCV:")
    LOOCV(X_, y_, a, 0, 'norm')
    df3 = pd.DataFrame()
    df3.insert(0, "Norm. Equations Model Parameters (No Regularization)", norm_theta, True)
    print(df3)

    # 2B
    reg_param2 = 6.5
    print("\n-----------------2B---------------\n")
    norm_theta2 = normal_equations(X_, y_, reg_param2)
    print("Training:")
    mpg_estimate(X_, y_, norm_theta2)
    print("LOOCV:")
    LOOCV(X_, y_, a, reg_param2, 'norm')
    df4 = pd.DataFrame()
    df4.insert(0, "Norm. Equations Model Parameters (Regularization)", norm_theta2, True)
    print(df4)


    # 2C

    # 2D
    ridge_trace_plot(X_, y_, a, 'norm')