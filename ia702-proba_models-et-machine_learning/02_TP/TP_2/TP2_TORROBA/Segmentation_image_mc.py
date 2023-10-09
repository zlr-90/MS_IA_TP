import numpy as np
import cv2
import scipy
import sklearn.cluster
import scipy.stats
import scipy.spatial.distance
import matplotlib.pyplot as plt
import pandas as pd
import utils
import math


def load_img_to_peano(img_name):
    return utils.peano_transform_img(cv2.cvtColor(cv2.imread('images/' + img_name), cv2.COLOR_BGR2GRAY))


# Partie III. Question 1)
def gauss2(Y, n, m1, sig1, m2, sig2):
    Y1 = scipy.stats.norm.pdf(Y, m1, sig1)
    Y2 = scipy.stats.norm.pdf(Y, m2, sig2)
    return np.hstack((Y1.reshape(-1, 1), Y2.reshape(-1, 1)))


# Partie III. Question 2)
def forward2(Mat_f, A, p10, p20):
    a1 = Mat_f[0, 0] * p10
    a2 = Mat_f[0, 1] * p20
    arr_a1 = [a1/(a1 + a2)]
    arr_a2 = [a2/(a1 + a2)]
    for i in range(1, Mat_f.shape[0]):
        a1 = Mat_f[i, 0] * (arr_a1[-1] * A[0, 0] + arr_a2[-1] * A[1, 0])
        a2 = Mat_f[i, 1] * (arr_a1[-1] * A[0, 1] + arr_a2[-1] * A[1, 1])
        arr_a1.append(a1/(a1 + a2))
        arr_a2.append(a2/(a1 + a2))
    arr_a1 = np.array(arr_a1)
    arr_a2 = np.array(arr_a2)
    return np.hstack((arr_a1.reshape(-1, 1), arr_a2.reshape(-1, 1)))


# Partie III. Question 3)
def backward2(Mat_f, A):
    arr_b1 = [0.5]
    arr_b2 = [0.5]
    for i in np.arange(Mat_f.shape[0] - 2, -1, -1):
        b1 = arr_b1[-1] * A[0, 0] * Mat_f[i + 1, 0] + arr_b2[-1] * A[0, 1] * Mat_f[i + 1, 1]
        b2 = arr_b1[-1] * A[1, 0] * Mat_f[i + 1, 0] + arr_b2[-1] * A[1, 1] * Mat_f[i + 1, 1]
        arr_b1.append(b1 / (b1 + b2))
        arr_b2.append(b2 / (b1 + b2))
    arr_b1 = np.array(arr_b1)
    arr_b2 = np.array(arr_b2)
    return np.flip(np.hstack((arr_b1.reshape(-1, 1), arr_b2.reshape(-1, 1))), 0)


# Partie III. Question 4)
def MPM_chaines2(Mat_f, n, cl1, cl2, A, p10, p20):
    alfa = forward2(Mat_f=Mat_f, A=A, p10=p10, p20=p20)
    beta = backward2(Mat_f=Mat_f, A=A)
    denominator = alfa[:, 0] * beta[:, 0] + alfa[:, 1] * beta[:, 1]
    predict1 = alfa[:, 0] * beta[:, 0] / denominator
    predict2 = alfa[:, 1] * beta[:, 1] / denominator
    return (predict1 >= predict2) * cl1 + (predict2 > predict1) * cl2


# Partie III. Question 5)
def calc_probaprio_mc(X, cl1, cl2):
    p = np.array([np.mean(X == cl1), 1 - np.mean(X == cl1)])
    A = pd.crosstab(pd.Series(X[:-1], name='Xn'), pd.Series(X[1:], name='Xn+1'), normalize=0)
    print(A)
    return p, A.values


# Partie III. Question 6)
def calc_next_psi_flatten(n, Mat_f, alfa, beta, A):
    extend_Mat_f = np.hstack((Mat_f, Mat_f))
    extend_alfa = np.hstack((alfa[:, 0].reshape(-1, 1), alfa[:, 0].reshape(-1, 1),
                             alfa[:, 1].reshape(-1, 1), alfa[:, 1].reshape(-1, 1)))
    extend_beta = np.hstack((beta, beta))
    psi = extend_alfa[:-1, :] * A.flatten() * extend_Mat_f[1:, :] * extend_beta[1:, :]
    return psi / np.sum(psi, axis=1).reshape(-1, 1)


# Partie III. Question 6)
def calc_next_ksi(alfa, beta):
    ksi = alfa * beta
    return ksi / np.sum(ksi, axis=1).reshape(-1, 1)


# Partie III. Question 6)
def calc_next_p(ksi):
    return np.mean(ksi, axis=0)


# Partie III. Question 6)
def calc_next_A(psi_flatten, ksi):
    return np.sum(psi_flatten, axis=0).reshape(2, 2) / np.sum(ksi[:-1, :], axis=0).reshape(-1, 1)


# Partie III. Question 6)
def calc_next_m(Y, ksi):
    return np.sum(Y.reshape(-1, 1) * ksi, axis=0) / np.sum(ksi, axis=0)


# Partie III. Question 6)
def calc_next_sig(Y, ksi, next_m_K):
    return np.sum(((np.hstack((Y.reshape(-1, 1), Y.reshape(-1, 1))) - next_m_K) ** 2) * ksi, axis=0) / np.sum(ksi, axis=0)


# Partie III. Question 6)
def estim_param_EM_mc(K, Y, A, p10, p20, m1, sig1, m2, sig2):
    A_K, p10_K, p20_K, m1_K, sig1_K, m2_K, sig2_K = A, p10, p20, m1, sig1, m2, sig2
    for k in range(K):
        Mat_f = gauss2(Y=Y, n=len(Y), m1=m1_K, sig1=sig1_K, m2=m2_K, sig2=sig2_K)
        alfa = forward2(Mat_f=Mat_f, A=A_K, p10=p10_K, p20=p20_K)
        beta = backward2(Mat_f=Mat_f, A=A_K)

        psi_flatten = calc_next_psi_flatten(n=Mat_f.shape[0], Mat_f=Mat_f, alfa=alfa, beta=beta, A=A_K)

        ksi = calc_next_ksi(alfa=alfa, beta=beta)

        p_K = calc_next_p(ksi=ksi)
        p10_K = p_K[0]
        p20_K = 1 - p_K[0]

        A_K = calc_next_A(psi_flatten=psi_flatten, ksi=ksi)
        print(A_K)
        print(np.sum(A_K, axis=1))
        m_K = calc_next_m(Y=Y, ksi=ksi)
        m1_K = m_K[0]
        m2_K = m_K[1]

        sig_K = np.sqrt(calc_next_sig(Y, ksi, next_m_K=m_K))
        sig1_K = sig_K[0]
        sig2_K = sig_K[1]
    return A_K, p10_K, p20_K, m1_K, sig1_K, m2_K, sig2_K


# Partie III. Question 7)
def calc_init_kmeans_mc(Y):
    k_means = sklearn.cluster.KMeans(n_clusters=2, random_state=0).fit(Y.reshape(-1, 1))
    m1, m2 = k_means.cluster_centers_[0][0], k_means.cluster_centers_[1][0]
    sq_dist = np.min(scipy.spatial.distance.cdist(Y.reshape(-1, 1), k_means.cluster_centers_), axis=1) ** 2
    var1 = np.sum((sq_dist * (k_means.labels_ == 0))) / np.sum(k_means.labels_ == 0)
    var2 = np.sum((sq_dist * (k_means.labels_ == 1))) / np.sum(k_means.labels_ == 1)
    sig1, sig2 = np.sqrt(var1), np.sqrt(var2)
    return k_means.labels_, m1, sig1, m2, sig2


# Partie III. Question 7)
def show_images_mc(img_name, m1, sig1, m2, sig2, X_img, Y_img, S_img):
    plt.figure(figsize=(10, 7))
    plt.suptitle(f'{img_name} | m1 : {m1}, sig1 : {sig1} ; m2 : {m2}, sig2 : {sig2}')
    plt.subplot(1, 3, 1)
    plt.title('Image originelle')
    plt.axis('off')
    plt.imshow(X_img, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 3, 2)
    plt.title('Image bruitée')
    plt.axis('off')
    plt.imshow(Y_img, cmap='gray', vmin=np.min(Y_img), vmax=np.max(Y_img))
    plt.subplot(1, 3, 3)
    plt.title('Image segmentée')
    plt.axis('off')
    plt.imshow(S_img, cmap='gray', vmin=0, vmax=255)
    plt.show(block=False)
    plt.pause(3)
    plt.close()


# Partie III. Question 7)
def segmentation_image_mc(img_name, m1, sig1, m2, sig2):
    print(f'-----------\n Traitement {img_name} | m1 : {m1}, sig1 : {sig1} ; m2 : {m2}, sig2 : {sig2}')

    X = load_img_to_peano(img_name)
    Y = utils.bruit_gauss2(X=X, cl1=0, cl2=255, m1=m1, sig1=sig1, m2=m2, sig2=sig2)

    labels, m1_0, sig1_0, m2_0, sig2_0 = calc_init_kmeans_mc(Y=Y)
    p_0, A_0 = calc_probaprio_mc(X=labels, cl1=0, cl2=1)
    A_K, p10_K, p20_K, m1_K, sig1_K, m2_K, sig2_K = estim_param_EM_mc(K=40, Y=Y, A=A_0, p10=p_0[0], p20=p_0[1], m1=m1_0,
                                                                      sig1=sig1_0, m2=m2_0, sig2=sig2_0)
    Mat_f = gauss2(Y=Y, n=len(Y), m1=m1_K, sig1=sig1_K, m2=m2_K, sig2=sig2_K)

    S = MPM_chaines2(Mat_f=Mat_f, n=Mat_f.shape[0], cl1=0, cl2=255, A=A_K, p10=p10_K, p20=p20_K)
    S_inv = (S == 0) * 255

    X_img = utils.transform_peano_in_img(X, int(math.sqrt(len(X))))
    Y_img = utils.transform_peano_in_img(Y, int(math.sqrt(len(Y))))
    if utils.taux_erreur(X, S) > utils.taux_erreur(X, S_inv):
        S = S_inv
    S_img = utils.transform_peano_in_img(S, int(math.sqrt(len(S))))

    show_images_mc(img_name=img_name, m1=m1, sig1=sig1, m2=m2, sig2=sig2, X_img=X_img, Y_img=Y_img, S_img=S_img)

    return utils.taux_erreur(X, S)
