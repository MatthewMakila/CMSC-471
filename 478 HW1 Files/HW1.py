# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import time
import openpyxl
import numpy as nump
import matplotlib.pyplot as plt


def file_import(f_name):
    # specify features and the response var for file extraction and storage as a table
    feat_names = ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
    f_feat_names = ['x0', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
    df = pd.read_excel(f_name, usecols=feat_names)
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

def gradient_descent(X, y, alpha):
    # create vector to hold the theta values (past and present), initialize our theta vectors to 0
    start = time.time()
    runtime = 0
    iteration_list = []
    iterations = 0
    threshold = 0.00001
    m = len(X[0])  # number of observations
    thetas = []
    old_thetas = []
    diff_vect = []
    costs = []
    for i in range(11):
        thetas.append(0)
        old_thetas.append(0)
        diff_vect.append(0)
    not_converged = True
    true_count = 0
    while not_converged:
        # loop over GDA equation algorithm
        for j in range(len(thetas)):    # loop to update each feature
            g_sum = 0
            for i in range(m):  # loop to compute each summation over all observations
                h_theta_x = 0
                for n in range(len(thetas)):
                    h_theta_x += (old_thetas[n] * X[n][i])
                g_sum += (X[j][i] * (h_theta_x - y[i]))
            # add back to new theta here
            thetas[j] = old_thetas[j] - (alpha / m) * g_sum
        # after one iteration, calculate the cost:
        g_sum = 0
        for i in range(m):  # loop to compute each summation over all observations
            h_theta_x = 0
            for n in range(len(thetas)):
                h_theta_x += (old_thetas[n] * X[n][i])
            g_sum += pow((h_theta_x - y[i]), 2)
        costs.append((1/(2*m)) * g_sum)
        iterations += 1
        iteration_list.append(iterations)
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
            print(thetas)
        else:
            # update old thetas to continue GDA next iteration
            true_count = 0
            for i in range(len(thetas)):
                old_thetas[i] = thetas[i]
    return thetas, runtime, costs, iteration_list


def plotting(xs, ys, rates):
    for i in range(len(xs)):
        plt.plot(xs[i][200:], ys[i][200:], label="alpha = {}".format(rates[i]))
    plt.legend()
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.show()


def plot_runtime(rt, rates):
    plt.grid()
    plt.scatter(rates, rt)
    plt.ylabel("Runtime")
    plt.xlabel("Learning Rate")
    plt.show()


def normal_equations(feat, resp):
    X1 = nump.array(feat).T
    y1 = nump.array(resp)
    temp = nump.dot(X1.T, X1)
    temp = nump.linalg.inv(temp)
    temp = nump.dot(temp, X1.T)
    temp = nump.dot(temp, y1)
    Thetas = temp
    print(Thetas)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X, y = file_import('mileage.xlsx')

    cost_vect = []
    iteration_vect = []
    runtime_vect = []
    alphas = [0.1, 0.15, 0.2, 0.25, 0.3]

    Thetas, runtime0, costs0, iterations0 = gradient_descent(X, y, 0.1)
    cost_vect.append(costs0)
    runtime_vect.append(runtime0)
    iteration_vect.append(iterations0)

    Thetas1, runtime1, costs1, iterations1 = gradient_descent(X, y, 0.15)
    cost_vect.append(costs1)
    runtime_vect.append(runtime1)
    iteration_vect.append(iterations1)

    Thetas2, runtime2, costs2, iterations2 = gradient_descent(X, y, 0.2)
    cost_vect.append(costs2)
    runtime_vect.append(runtime2)
    iteration_vect.append(iterations2)

    Thetas3, runtime3, costs3, iterations3 = gradient_descent(X, y, 0.25)
    cost_vect.append(costs3)
    runtime_vect.append(runtime3)
    iteration_vect.append(iterations3)

    Thetas4, runtime4, costs4, iterations4 = gradient_descent(X, y, 0.3)
    cost_vect.append(costs4)
    runtime_vect.append(runtime4)
    iteration_vect.append(iterations4)

    # create normal equations theta outputs
    normal_equations(X, y)
    # create plot of cost functions vs. iterations
    plotting(iteration_vect, cost_vect, alphas)
    # plot of runtimes vs. alphas
    plot_runtime(runtime_vect, alphas)

