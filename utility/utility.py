import os
import pathlib

import numpy as np
import scipy.io as scio
from scipy.stats import scoreatpercentile
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler

from models.lof import Lof
from models.knn import Knn


def get_label_n(y, y_pred):
    out_perc = np.count_nonzero(y) / len(y)
    threshold = scoreatpercentile(y_pred, 100 * (1 - out_perc))
    y_pred = (y_pred > threshold).astype('int')
    return y_pred


def standardizer(X_train, X_test):
    '''
    normalization function wrapper
    :param X_train:
    :param X_test:
    :return: X_train and X_test after the Z-score normalization
    '''
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def precision_n_score(y, y_pred):
    '''
    Utlity function to calculate precision@n
    :param y: ground truth
    :param y_pred: number of outliers
    :return: score
    '''
    # calculate the percentage of outliers
    out_perc = np.count_nonzero(y) / len(y)

    threshold = scoreatpercentile(y_pred, 100 * (1 - out_perc))
    y_pred = (y_pred > threshold).astype('int')
    return precision_score(y, y_pred)


def argmaxp(a, p):
    '''
    Utlity function to return the index of top p values in a
    :param a: list variable
    :param p: number of elements to select
    :return: index of top p elements in a
    '''

    a = np.asarray(a).ravel()
    length = a.shape[0]
    pth = np.argpartition(a, length - p)
    return pth[length - p:]


def loaddata(filename):
    '''
    load data
    :param filename:
    :return:
    '''
    mat = scio.loadmat(os.path.join('datasets', filename + '.mat'))
    X_orig = mat['X']
    y_orig = mat['y'].ravel()
    outlier_perc = np.count_nonzero(y_orig) / len(y_orig)

    return X_orig, y_orig, outlier_perc


def train_predict_lof(k_list, X_train_norm, X_test_norm, train_scores,
                      test_scores):
    # initialize base detectors
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

    return train_scores, test_scores


def train_predict_knn(k_list, X_train_norm, X_test_norm, train_scores,
                      test_scores):
    # initialize base detectors
    clf_list = []
    for k in k_list:
        clf = Knn(n_neighbors=k, method='largest')
        clf.fit(X_train_norm)
        train_score = clf.decision_scores
        test_score = clf.decision_function(X_test_norm)
        clf_name = 'knn_' + str(k)

        clf_list.append(clf_name)
        curr_ind = len(clf_list) - 1

        train_scores[:, curr_ind] = train_score.ravel()
        test_scores[:, curr_ind] = test_score.ravel()

    return train_scores, test_scores


def print_save_result(data, base_detector, n_baselines, n_clf, n_ite, roc_mat,
                      ap_mat, prc_mat, method_list, timestamp, verbose):
    '''
    :param data:
    :param base_detector:
    :param n_baselines:
    :param n_clf:
    :param n_ite:
    :param roc_mat:
    :param ap_mat:
    :param prc_mat:
    :param method_list:
    :param timestamp:
    :param verbose:
    :return: None
    '''

    roc_scores = np.round(np.mean(roc_mat, axis=0), decimals=4)
    ap_scores = np.round(np.mean(ap_mat, axis=0), decimals=4)
    prc_scores = np.round(np.mean(prc_mat, axis=0), decimals=4)

    method_np = np.asarray(method_list)

    top_roc_ind = argmaxp(roc_scores, 1)
    top_ap_ind = argmaxp(ap_scores, 1)
    top_prc_ind = argmaxp(prc_scores, 1)

    top_roc_clf = method_np[top_roc_ind].tolist()[0]
    top_ap_clf = method_np[top_ap_ind].tolist()[0]
    top_prc_clf = method_np[top_prc_ind].tolist()[0]

    top_roc = np.round(roc_scores[top_roc_ind][0], decimals=4)
    top_ap = np.round(ap_scores[top_ap_ind][0], decimals=4)
    top_prc = np.round(prc_scores[top_prc_ind][0], decimals=4)

    roc_diff = np.round(100 * (top_roc - roc_scores) / roc_scores, decimals=4)
    ap_diff = np.round(100 * (top_ap - ap_scores) / ap_scores, decimals=4)
    prc_diff = np.round(100 * (top_prc - prc_scores) / prc_scores, decimals=4)

    # initialize the log directory if it does not exist
    pathlib.Path('results').mkdir(parents=True, exist_ok=True)

    # create the file if it does not exist
    f = open(
        'results\\' + data + '_' + base_detector + '_' + timestamp + '.csv',
        'a')
    if verbose:
        f.writelines('method, '
                     'roc, best_roc, diff_roc,'
                     'ap, best_ap, diff_ap,'
                     'p@m, best_p@m, diff_p@m,'
                     'best roc, best ap, best prc, n_ite, n_clf')
    else:
        f.writelines('method, '
                     'roc, ap, p@m,'
                     'best roc, best ap, best prc, '
                     'n_ite, n_clf')

    print('method, roc, ap, p@m, best roc, best ap, best prc')
    delim = ','
    for i in range(n_baselines):
        print(method_list[i], roc_scores[i], ap_scores[i], prc_scores[i],
              top_roc_clf, top_ap_clf, top_prc_clf)

        if verbose:
            f.writelines(
                '\n' + str(method_list[i]) + delim +
                str(roc_scores[i]) + delim + str(top_roc) + delim + str(
                    roc_diff[i]) + delim +
                str(ap_scores[i]) + delim + str(top_ap) + delim + str(
                    ap_diff[i]) + delim +
                str(prc_scores[i]) + delim + str(top_prc) + delim + str(
                    prc_diff[i]) + delim +
                top_roc_clf + delim +
                top_ap_clf + delim +
                top_prc_clf + delim +
                str(n_ite) + delim +
                str(n_clf))
        else:
            f.writelines(
                '\n' + str(method_list[i]) + delim +
                str(roc_scores[i]) + delim +
                str(ap_scores[i]) + delim +
                str(prc_scores[i]) + delim +
                top_roc_clf + delim +
                top_ap_clf + delim +
                top_prc_clf + delim +
                str(n_ite) + delim +
                str(n_clf))

    f.close()
