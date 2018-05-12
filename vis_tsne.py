import math
import pathlib
import numpy as np
from scipy.stats import rankdata
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree

from sklearn.manifold import TSNE

from models.combination import aom, moa
from models.lof import Lof
from utility.stat_models import wpearsonr
from utility.utility import argmaxp, loaddata, standardizer, get_label_n

# set numpy parameters
np.set_printoptions(suppress=True, precision=4)
# generates the visualization for all datasets
data_list = ["Annthyroid",
             "Pendigits",
             "Satellite",
             "Pima",
             "Letter",
             "Thyroid",
             "Vowels",
             "Cardio",
             "Mnist"]
dodc_best_list = [186, 38, 71, 103, 233, 157, 128, 127, 97]

for data, dodc_best in zip(data_list, dodc_best_list):

    print('processing', data)
    X_test_list = []
    X_test_name_list = []
    dodc_best_list = []
    test_target_list_list = []
    y_test_list = []
    trans_data_list = []

    X_orig, y_orig, outlier_perc = loaddata(data)

    ite = 1  # number of iterations
    test_size = 0.4  # training = 60%, testing = 40%
    final_k_list = [10, 30, 60, 100]
    n_methods = 253

    k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
              110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

    n_clf = len(k_list)
    fixed_range = [5, 10, 15]

    # for AOM and MOA, choose the right number of buckets
    n_buckets = 5
    n_clf_bucket = int(n_clf / n_buckets)
    assert (n_clf % n_buckets == 0)  # in case wrong number of buckets

    # split the data into training and testing
    # fixed the visualization by random state == 42
    X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig,
                                                        test_size=test_size,
                                                        random_state=42)

    # generate the normalized data
    X_train_norm, X_test_norm = standardizer(X_train, X_test)

    train_scores = np.zeros([X_train.shape[0], n_clf])
    test_scores = np.zeros([X_test.shape[0], n_clf])

    # initialized the list to store the results
    test_target_list = []
    method_list = []
    k_rec_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # zeros for non dcs

    clf_list = []
    for k in k_list:
        clf = Lof(n_neighbors=k)
        clf.fit(X_train_norm)
        train_score = clf.negative_outlier_factor_ * -1
        test_score = clf.decision_function(X_test_norm) * -1
        clf_name = 'lof_' + str(k)

        clf_list.append(clf_name)
        curr_ind = len(clf_list) - 1

        train_scores[:, curr_ind] = train_score.ravel()
        test_scores[:, curr_ind] = test_score.ravel()

    #######################################################################
    # generate normalized scores
    train_scores_norm, test_scores_norm = standardizer(train_scores,
                                                       test_scores)

    # make sure the scores are actually standardized
    assert (math.isclose(train_scores_norm.mean(), 0, abs_tol=0.1))
    #    assert(math.isclose(test_scores_norm.mean(), 0, abs_tol=0.1))
    assert (math.isclose(train_scores_norm.std(), 1, abs_tol=0.1))
    #    assert(math.isclose(test_scores_norm.std(), 1, abs_tol=0.1))

    # generate mean and max outputs
    target_test_mean = np.mean(test_scores_norm, axis=1)
    target_test_max = np.max(test_scores_norm, axis=1)
    test_target_list.extend([target_test_mean, target_test_max])
    method_list.extend(['mean', 'max'])

    # generate pseudo target for training -> for calculating weights
    target_mean = np.mean(train_scores_norm, axis=1).reshape(-1, 1)
    target_max = np.max(train_scores_norm, axis=1).reshape(-1, 1)
    # higher value for more outlierness
    ranks_mean = rankdata(target_mean).reshape(-1, 1)
    ranks_max = rankdata(target_max).reshape(-1, 1)

    # generate weighted mean
    # weights are distance or pearson in different modes
    clf_weights_pear = np.zeros([n_clf, 1])
    for i in range(n_clf):
        clf_weights_pear[i] = \
            pearsonr(target_mean, train_scores_norm[:, i].reshape(-1, 1))[0][0]

    clf_weights_euc = np.zeros([n_clf, 1])
    for i in range(n_clf):
        clf_weights_euc[i] = euclidean(target_mean,
                                       train_scores_norm[:, i].reshape(-1, 1))
    clf_weights_euc = clf_weights_euc.max() - clf_weights_euc

    for i in fixed_range:
        target_test_max_pear = np.max(
            test_scores_norm[:, argmaxp(clf_weights_pear, i)], axis=1)
        target_test_max_euc = np.max(
            test_scores_norm[:, argmaxp(clf_weights_euc, i)], axis=1)
        test_target_list.extend([target_test_max_pear, target_test_max_euc])
        method_list.extend(
            ['max_' + str(i) + '_pear', 'max_' + str(i) + '_euc'])

    # generate weighted mean
    target_test_weighted_pear = np.sum(
        test_scores_norm * clf_weights_pear.reshape(1,
                                                    -1) / clf_weights_pear.sum(),
        axis=1)
    target_test_weighted_euc = np.sum(
        test_scores_norm * clf_weights_euc.reshape(1,
                                                   -1) / clf_weights_euc.sum(),
        axis=1)
    test_target_list.extend(
        [target_test_weighted_pear, target_test_weighted_euc])
    method_list.extend(['w_mean_pear', 'w_mean_euc', ])

    # generate threshold sum
    target_test_threshold = np.sum(test_scores_norm.clip(0), axis=1)
    test_target_list.append(target_test_threshold)
    method_list.append('threshold')

    # generate average of maximum (AOM) and maximum of average (MOA)
    target_test_aom = aom(test_scores_norm, n_buckets, n_clf)
    target_test_moa = moa(test_scores_norm, n_buckets, n_clf)
    test_target_list.extend([target_test_aom, target_test_moa])
    method_list.extend(['aom', 'moa'])
    ###################################################################
    # use mean as the pseudo target
    for k in final_k_list:
        tree = KDTree(X_train_norm)
        dist_arr, ind_arr = tree.query(X_test_norm, k=k)

        m_list = ['a_dist_d', 'a_dist_r', 'a_dist_n',
                  'a_pear_d', 'a_pear_r', 'a_pear_n']

        # initialize different buckets
        pred_scores_best = np.zeros([X_test.shape[0], len(m_list)])
        pred_scores_max_d = np.zeros([X_test.shape[0], len(m_list)])
        pred_scores_max_f5 = np.zeros([X_test.shape[0], len(m_list)])
        pred_scores_max_f10 = np.zeros([X_test.shape[0], len(m_list)])
        pred_scores_max_f15 = np.zeros([X_test.shape[0], len(m_list)])

        for i in range(X_test.shape[0]):  # X_test_norm.shape[0]
            # get the neighbor idx of the current point
            ind_k = ind_arr[i, :]

            # get the pseudo target: mean
            target_k = target_mean[ind_k,].ravel()

            # get the current scores from all clf
            curr_train_k = train_scores_norm[ind_k, :]

            # weights by rank
            weights_k_rank = ranks_mean[ind_k]

            # weights by distance
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
                corr_dist_n[d,] = euclidean(target_k, curr_train_k[:, d]) * -1
                corr_pear_d[d,] = wpearsonr(target_k, curr_train_k[:, d],
                                            w=weights_k_dist)
                corr_pear_r[d,] = wpearsonr(target_k, curr_train_k[:, d],
                                            w=weights_k_rank)
                corr_pear_n[d,] = wpearsonr(target_k, curr_train_k[:, d])[0]

            corr_list = [corr_dist_d, corr_dist_r, corr_dist_n,
                         corr_pear_d, corr_pear_r, corr_pear_n]

            for j in range(len(m_list)):
                corr_k = corr_list[j]

                # pick the best one
                best_clf_ind = np.nanargmax(corr_k)
                pred_scores_best[i, j] = test_scores_norm[i, best_clf_ind]
                #                print(k, best_clf_ind)
                # pick the p dynamically
                threshold = corr_k.max() - corr_k.std() * 0.2
                p = (corr_k >= threshold).sum()
                if p == 0:  # in case extreme cases [nan and all -1's]
                    p = 1
                pred_scores_max_d[i, j] = np.max(
                    test_scores_norm[i, argmaxp(corr_k, p)])

                # pick the best 5 classifiers
                pred_scores_max_f5[i, j] = np.max(
                    test_scores_norm[i, argmaxp(corr_k, 5)])
                # pick the best 10 classifiers
                pred_scores_max_f10[i, j] = np.max(
                    test_scores_norm[i, argmaxp(corr_k, 10)])
                # pick the best 15 classifiers
                pred_scores_max_f15[i, j] = np.max(
                    test_scores_norm[i, argmaxp(corr_k, 15)])

        for m in range(len(m_list)):
            test_target_list.extend([pred_scores_best[:, m],
                                     pred_scores_max_d[:, m],
                                     pred_scores_max_f5[:, m],
                                     pred_scores_max_f10[:, m],
                                     pred_scores_max_f15[:, m]])
            method_list.extend(['dcs_best_' + m_list[m] + '_' + str(k),
                                'dcs_dyn_' + m_list[m] + '_' + str(k),
                                'dcs_f5_' + m_list[m] + '_' + str(k),
                                'dcs_f10_' + m_list[m] + '_' + str(k),
                                'dcs_f15_' + m_list[m] + '_' + str(k)])
            k_rec_list.extend([k, k, k, k, k])
    ##########################################################################

    # use max for pseudo target
    for k in final_k_list:
        print('processing', k)
        tree = KDTree(X_train_norm)
        dist_arr, ind_arr = tree.query(X_test_norm, k=k)

        m_list = ['m_dist_d', 'm_dist_r', 'm_dist_n',
                  'm_pear_d', 'm_pear_r', 'm_pear_n']

        pred_scores_best = np.zeros([X_test.shape[0], len(m_list)])
        pred_scores_max_d = np.zeros([X_test.shape[0], len(m_list)])
        pred_scores_max_f5 = np.zeros([X_test.shape[0], len(m_list)])
        pred_scores_max_f10 = np.zeros([X_test.shape[0], len(m_list)])
        pred_scores_max_f15 = np.zeros([X_test.shape[0], len(m_list)])

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
                corr_dist_n[d,] = euclidean(target_k, curr_train_k[:, d]) * -1
                corr_pear_d[d,] = wpearsonr(target_k, curr_train_k[:, d],
                                            w=weights_k_dist)
                corr_pear_r[d,] = wpearsonr(target_k, curr_train_k[:, d],
                                            w=weights_k_rank)
                corr_pear_n[d,] = wpearsonr(target_k, curr_train_k[:, d])[0]

            corr_list = [corr_dist_d, corr_dist_r, corr_dist_n,
                         corr_pear_d, corr_pear_r, corr_pear_n]

            for j in range(len(m_list)):
                corr_k = corr_list[j]

                # pick the best one
                best_clf_ind = np.nanargmax(corr_k)
                pred_scores_best[i, j] = test_scores_norm[i, best_clf_ind]

                # pick the p dynamically
                threshold = corr_k.max() - corr_k.std() * 0.2
                p = (corr_k >= threshold).sum()
                if p == 0:  # in case extreme cases [nan and all -1's]
                    p = 1
                pred_scores_max_d[i, j] = np.mean(
                    test_scores_norm[i, argmaxp(corr_k, p)])

                # pick the best 5 classifiers
                pred_scores_max_f5[i, j] = np.mean(
                    test_scores_norm[i, argmaxp(corr_k, 5)])
                # pick the best 10 classifiers
                pred_scores_max_f10[i, j] = np.mean(
                    test_scores_norm[i, argmaxp(corr_k, 10)])
                # pick the best 15 classifiers
                pred_scores_max_f15[i, j] = np.mean(
                    test_scores_norm[i, argmaxp(corr_k, 15)])

        for m in range(len(m_list)):
            test_target_list.extend([pred_scores_best[:, m],
                                     pred_scores_max_d[:, m],
                                     pred_scores_max_f5[:, m],
                                     pred_scores_max_f10[:, m],
                                     pred_scores_max_f15[:, m]])
            method_list.extend(['dcs_best_' + m_list[m] + '_' + str(k),
                                'dcs_dyn_' + m_list[m] + '_' + str(k),
                                'dcs_f5_' + m_list[m] + '_' + str(k),
                                'dcs_f10_' + m_list[m] + '_' + str(k),
                                'dcs_f15_' + m_list[m] + '_' + str(k)])
            k_rec_list.extend([k, k, k, k, k])

    trans_data_list.append(
        TSNE(n_components=2, init='pca').fit_transform(X_test))
    X_test_list.append(X_test)
    X_test_name_list.append(data)
    dodc_best_list.append(dodc_best)
    test_target_list_list.append(test_target_list)
    y_test_list.append(y_test)
    ##########################################################################
    plt.figure(figsize=(12, 6))

    for k in range(1):

        # find the comparision
        dcs_target = get_label_n(y_test_list[k],
                                 test_target_list_list[k][dodc_best_list[k]])
        mean_target = get_label_n(y_test_list[k], test_target_list_list[k][0])
        max_target = get_label_n(y_test_list[k], test_target_list_list[k][1])

        normal_ind = []
        outlier_ind = []

        equal_right_mean = []
        equal_wrong_mean = []

        equal_right_max = []
        equal_wrong_max = []

        dcs_out_mean = []
        mean_out = []

        dcs_norm_mean = []
        mean_norm = []

        dcs_out_max = []
        max_out = []

        dcs_norm_max = []
        max_norm = []

        for i in range(X_test_list[k].shape[0]):
            if y_test_list[k][i] == 0:
                normal_ind.append(i)
            else:
                outlier_ind.append(i)

            if dcs_target[i] == mean_target[i] == y_test_list[k][i]:
                print(i, 'equal & right')
                equal_right_mean.append(i)

            elif dcs_target[i] == mean_target[i] and dcs_target[i] != \
                    y_test_list[k][i]:
                print(i, 'equal & wrong')
                equal_wrong_mean.append(i)

            elif dcs_target[i] != mean_target[i]:
                print(i, 'not equal')
                if y_test_list[k][i] == 1:
                    if dcs_target[i] == y_test_list[k][i]:
                        dcs_out_mean.append(i)
                    else:
                        mean_out.append(i)
                else:
                    if dcs_target[i] == y_test_list[k][i]:
                        dcs_norm_mean.append(i)
                    else:
                        mean_norm.append(i)
            ##################################################################
            if dcs_target[i] == max_target[i] == y_test_list[k][i]:
                print(i, 'equal & right')
                equal_right_max.append(i)

            elif dcs_target[i] == max_target[i] and dcs_target[i] != \
                    y_test_list[k][i]:
                print(i, 'equal & wrong')
                equal_wrong_max.append(i)

            elif dcs_target[i] != max_target[i]:
                print(i, 'not equal')
                if y_test_list[k][i] == 1:
                    if dcs_target[i] == y_test_list[k][i]:
                        dcs_out_max.append(i)
                    else:
                        max_out.append(i)
                else:
                    if dcs_target[i] == y_test_list[k][i]:
                        dcs_norm_max.append(i)
                    else:
                        max_norm.append(i)

        # plot mean
        plt.subplot(121)

        plt.scatter(trans_data_list[k][normal_ind, 0],
                    trans_data_list[k][normal_ind, 1], label='Normal',
                    color='orange', alpha=0.6, s=24, marker='o')
        plt.scatter(trans_data_list[k][outlier_ind, 0],
                    trans_data_list[k][outlier_ind, 1], label='Outlying',
                    color='red', alpha=0.6, s=28, marker='s')

        plt.scatter(trans_data_list[k][mean_norm, 0],
                    trans_data_list[k][mean_norm, 1], label='SG_N',
                    color='g', alpha=0.95, s=40, marker='v')
        plt.scatter(trans_data_list[k][mean_out, 0],
                    trans_data_list[k][mean_out, 1], label='SG_O',
                    color='g', alpha=0.95, s=40, marker='^')

        plt.scatter(trans_data_list[k][dcs_norm_max, 0],
                    trans_data_list[k][dcs_norm_max, 1], label='DODC_N',
                    color='b', alpha=0.95, s=54, marker='x')
        plt.scatter(trans_data_list[k][dcs_out_max, 0],
                    trans_data_list[k][dcs_out_max, 1], label='DODC_O',
                    color='b', alpha=0.95, s=65, marker='+')


        plt.legend(ncol=3, prop={'size': 7.5}, loc='lower right',
                   bbox_transform=plt.gcf().transFigure)
        plt.xticks([])
        plt.yticks([])
        plt.title('SG_A vs. DODC (' + X_test_name_list[k] + ')', fontsize=12)

        # plot max
        plt.subplot(122)

        plt.scatter(trans_data_list[k][normal_ind, 0],
                    trans_data_list[k][normal_ind, 1], label='Normal',
                    color='orange', alpha=0.6, s=24, marker='o')
        plt.scatter(trans_data_list[k][outlier_ind, 0],
                    trans_data_list[k][outlier_ind, 1], label='Outlying',
                    color='red', alpha=0.6, s=28, marker='s')

        plt.scatter(trans_data_list[k][max_norm, 0],
                    trans_data_list[k][max_norm, 1], label='SG_N',
                    color='g', alpha=0.95, s=40, marker='v')
        plt.scatter(trans_data_list[k][max_out, 0],
                    trans_data_list[k][max_out, 1], label='SG_O',
                    color='g', alpha=0.95, s=40, marker='^')

        plt.scatter(trans_data_list[k][dcs_norm_max, 0],
                    trans_data_list[k][dcs_norm_max, 1], label='DODC_N',
                    color='b', alpha=0.95, s=54, marker='x')
        plt.scatter(trans_data_list[k][dcs_out_max, 0],
                    trans_data_list[k][dcs_out_max, 1], label='DODC_O',
                    color='b', alpha=0.95, s=65, marker='+')

        plt.legend(ncol=3, prop={'size': 7.5}, loc='lower right',
                   bbox_transform=plt.gcf().transFigure)
        plt.xticks([])
        plt.yticks([])
        plt.title('SG_M vs. DODC (' + X_test_name_list[k] + ')', fontsize=12)

    plt.tight_layout()

    # initialize the log directory if it does not exist
    pathlib.Path('viz').mkdir(parents=True, exist_ok=True)
    # save files
    plt.savefig('viz\\' + data + '.png', dpi=330)
