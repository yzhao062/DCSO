import datetime

import numpy as np
from scipy.stats import rankdata
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from models.combination import aom, moa
from utility.stat_models import wpearsonr
from utility.utility import train_predict_knn
from utility.utility import print_save_result
from utility.utility import argmaxp, loaddata, precision_n_score, standardizer

# access the timestamp for logging purpose
today = datetime.datetime.now()
timestamp = today.strftime("%Y%m%d_%H%M%S")

# set numpy parameters
np.set_printoptions(suppress=True, precision=4)

###############################################################################
# parameter settings
data = 'cardio'
base_detector = 'knn'
n_ite = 20  # number of iterations
test_size = 0.4  # training = 60%, testing = 40%
n_baselines = 30  # the number of baseline algorithms, DO NOT CHANGE
loc_region_size = 100  # for consistency fixed to 100

# k list for LOF algorithms, for constructing a pool of base detectors
k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
          110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

n_clf = len(k_list)  # 20 base detectors

# for SG_AOM and SG_MOA, choose the right number of buckets
n_buckets = 5
n_clf_bucket = int(n_clf / n_buckets)
assert (n_clf % n_buckets == 0)  # in case wrong number of buckets

alpha = 0.2  # control the strength of dynamic ensemble selection

# flag for printing and output saving
verbose = False

###############################################################################

if __name__ == '__main__':

    X_orig, y_orig, outlier_perc = loaddata(data)

    # initialize the matrix for storing scores
    roc_mat = np.zeros([n_ite, n_baselines])  # receiver operating curve
    ap_mat = np.zeros([n_ite, n_baselines])  # average precision
    prc_mat = np.zeros([n_ite, n_baselines])  # precision @ m

    for t in range(n_ite):
        print('\nn_ite', t + 1, data)  # print status

        # split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig,
                                                            test_size=test_size)
        # normalized the data
        X_train_norm, X_test_norm = standardizer(X_train, X_test)

        train_scores = np.zeros([X_train.shape[0], n_clf])
        test_scores = np.zeros([X_test.shape[0], n_clf])

        # initialized the list to store the results
        test_target_list = []
        method_list = []

        # generate a pool of detectors and predict on test instances
        train_scores, test_scores = train_predict_knn(k_list, X_train_norm,
                                                      X_test_norm,
                                                      train_scores,
                                                      test_scores)

        #######################################################################
        # generate normalized scores
        train_scores_norm, test_scores_norm = standardizer(train_scores,
                                                           test_scores)
        # generate mean and max outputs
        # SG_A and SG_M
        target_test_mean = np.mean(test_scores_norm, axis=1)
        target_test_max = np.max(test_scores_norm, axis=1)
        test_target_list.extend([target_test_mean, target_test_max])
        method_list.extend(['sg_a', 'sg_m'])

        # generate pseudo target for training -> for calculating weights
        target_mean = np.mean(train_scores_norm, axis=1).reshape(-1, 1)
        target_max = np.max(train_scores_norm, axis=1).reshape(-1, 1)

        # higher value for more outlyingness
        ranks_mean = rankdata(target_mean).reshape(-1, 1)
        ranks_max = rankdata(target_max).reshape(-1, 1)

        # generate weighted mean
        # weights are distance or pearson in different modes
        clf_weights_pear = np.zeros([n_clf, 1])
        for i in range(n_clf):
            clf_weights_pear[i] = \
                pearsonr(target_mean, train_scores_norm[:, i].reshape(-1, 1))[
                    0][0]

        # generate weighted mean
        target_test_weighted_pear = np.sum(
            test_scores_norm * clf_weights_pear.reshape(1,
                                                        -1) / clf_weights_pear.sum(),
            axis=1)

        test_target_list.append(target_test_weighted_pear)
        method_list.append('sg_wa')

        # generate threshold sum
        target_test_threshold = np.sum(test_scores_norm.clip(0), axis=1)
        test_target_list.append(target_test_threshold)
        method_list.append('sg_thresh')

        # generate average of maximum (SG_AOM) and maximum of average (SG_MOA)
        target_test_aom = aom(test_scores_norm, n_buckets, n_clf)
        target_test_moa = moa(test_scores_norm, n_buckets, n_clf)
        test_target_list.extend([target_test_aom, target_test_moa])
        method_list.extend(['aom', 'moa'])
        ##################################################################

        # define local region using KD trees
        tree = KDTree(X_train_norm)
        dist_arr, ind_arr = tree.query(X_test_norm, k=loc_region_size)

        # different similarity measures
        # s[euc]_w[rank] -> use euclidean distance for similarity measure
        #                   use outlying rank as the weight
        m_list = ['s[euc]_w[dist]', 's[euc]_w[rank]', 's[dist]_w[na]',
                  's[pear]_w[dist]', 's[pear]_w[rank]', 's[pear]_w[na]']

        pred_scores_best = np.zeros([X_test.shape[0], len(m_list)])
        pred_scores_ens = np.zeros([X_test.shape[0], len(m_list)])

        for i in range(X_test.shape[0]):  # iterate all test instance
            # get the neighbor idx of the current point
            ind_k = ind_arr[i, :]

            # get the pseudo target: mean
            target_k = target_mean[ind_k,].ravel()

            # get the current scores from all clf
            curr_train_k = train_scores_norm[ind_k, :]

            # weights by rank
            weights_k_rank = ranks_mean[ind_k]

            # weights by euclidean distance
            dist_k = dist_arr[i, :].reshape(-1, 1)
            weights_k_dist = dist_k.max() - dist_k

            # initialize containers for correlation
            corr_dist_d = np.zeros([n_clf, ])
            corr_dist_r = np.zeros([n_clf, ])
            corr_dist_n = np.zeros([n_clf, ])
            corr_pear_d = np.zeros([n_clf, ])
            corr_pear_r = np.zeros([n_clf, ])
            corr_pear_n = np.zeros([n_clf, ])

            for d in range(n_clf):
                # flip distance so larger values imply larger correlation
                corr_dist_d[d,] = euclidean(target_k, curr_train_k[:, d],
                                            w=weights_k_dist) * -1
                corr_dist_r[d,] = euclidean(target_k, curr_train_k[:, d],
                                            w=weights_k_rank) * -1
                corr_dist_n[d,] = euclidean(target_k,
                                            curr_train_k[:, d]) * -1
                corr_pear_d[d,] = wpearsonr(target_k, curr_train_k[:, d],
                                            w=weights_k_dist)
                corr_pear_r[d,] = wpearsonr(target_k, curr_train_k[:, d],
                                            w=weights_k_rank)
                corr_pear_n[d,] = wpearsonr(target_k, curr_train_k[:, d])[
                    0]

            corr_list = [corr_dist_d, corr_dist_r, corr_dist_n,
                         corr_pear_d, corr_pear_r, corr_pear_n]

            for j in range(len(m_list)):
                corr_k = corr_list[j]

                # pick the best one
                best_clf_ind = np.nanargmax(corr_k)
                pred_scores_best[i, j] = test_scores_norm[i, best_clf_ind]

                # pick the p dynamically
                threshold = corr_k.max() - corr_k.std() * alpha
                p = (corr_k >= threshold).sum()
                if p == 0:  # in case extreme cases [nan and all -1's]
                    p = 1
                pred_scores_ens[i, j] = np.max(
                    test_scores_norm[i, argmaxp(corr_k, p)])

        for m in range(len(m_list)):
            test_target_list.extend([pred_scores_best[:, m],
                                     pred_scores_ens[:, m]])
            method_list.extend(['dodc_a_' + m_list[m],
                                'dodc_moa_' + m_list[m]])
        ######################################################################

        # use max for pseudo ground truth generation
        tree = KDTree(X_train_norm)
        dist_arr, ind_arr = tree.query(X_test_norm, k=loc_region_size)

        pred_scores_best = np.zeros([X_test.shape[0], len(m_list)])
        pred_scores_ens = np.zeros([X_test.shape[0], len(m_list)])

        for i in range(X_test.shape[0]):  # X_test_norm.shape[0]
            # get the neighbor idx of the current point
            ind_k = ind_arr[i, :]

            # get the pseudo target: max
            target_k = target_max[ind_k,].ravel()

            # get the current scores from all clf
            curr_train_k = train_scores_norm[ind_k, :]

            # weights by rank
            weights_k_rank = ranks_max[ind_k]
            # weights by distance
            dist_k = dist_arr[i, :].reshape(-1, 1)
            weights_k_dist = dist_k.max() - dist_k

            corr_dist_d = np.zeros([n_clf, ])
            corr_dist_r = np.zeros([n_clf, ])
            corr_dist_n = np.zeros([n_clf, ])
            corr_pear_d = np.zeros([n_clf, ])
            corr_pear_r = np.zeros([n_clf, ])
            corr_pear_n = np.zeros([n_clf, ])

            for d in range(n_clf):
                corr_dist_d[d,] = euclidean(target_k, curr_train_k[:, d],
                                            w=weights_k_dist) * -1
                corr_dist_r[d,] = euclidean(target_k, curr_train_k[:, d],
                                            w=weights_k_rank) * -1
                corr_dist_n[d,] = euclidean(target_k,
                                            curr_train_k[:, d]) * -1
                corr_pear_d[d,] = wpearsonr(target_k, curr_train_k[:, d],
                                            w=weights_k_dist)
                corr_pear_r[d,] = wpearsonr(target_k, curr_train_k[:, d],
                                            w=weights_k_rank)
                corr_pear_n[d,] = wpearsonr(target_k, curr_train_k[:, d])[
                    0]

            corr_list = [corr_dist_d, corr_dist_r, corr_dist_n,
                         corr_pear_d, corr_pear_r, corr_pear_n]

            for j in range(len(m_list)):
                corr_k = corr_list[j]

                # pick the best one
                best_clf_ind = np.nanargmax(corr_k)
                pred_scores_best[i, j] = test_scores_norm[i, best_clf_ind]

                # pick s detectors dynamically
                threshold = corr_k.max() - corr_k.std() * alpha
                p = (corr_k >= threshold).sum()
                if p == 0:  # in case extreme cases [nan and all -1's]
                    p = 1
                pred_scores_ens[i, j] = np.mean(
                    test_scores_norm[i, argmaxp(corr_k, p)])

        for m in range(len(m_list)):
            test_target_list.extend([pred_scores_best[:, m],
                                     pred_scores_ens[:, m]])
            method_list.extend(['dodc_m_' + m_list[m],
                                'dodc_aom_' + m_list[m]])

        # store performance information and print result
        for i in range(n_baselines):
            roc_mat[t, i] = roc_auc_score(y_test, test_target_list[i])
            ap_mat[t, i] = average_precision_score(y_test,
                                                   test_target_list[i])
            prc_mat[t, i] = precision_n_score(y_test, test_target_list[i])
            print(method_list[i], roc_mat[t, i])

    # print and save the result
    # default location is /results/***.csv
    print_save_result(data, base_detector, n_baselines, n_clf, n_ite, roc_mat,
                      ap_mat, prc_mat, method_list, timestamp, verbose)
