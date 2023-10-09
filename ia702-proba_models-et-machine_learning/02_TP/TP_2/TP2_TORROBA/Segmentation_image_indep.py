import numpy as np
import cv2
import scipy
import sklearn.cluster
import scipy.stats
import scipy.spatial.distance
import matplotlib.pyplot as plt
import utils
import math


# Partie II. Question 4)
def load_img_to_line(img_name):
    return utils.line_transform_img(cv2.cvtColor(cv2.imread('images/' + img_name), cv2.COLOR_BGR2GRAY))


# Partie II. Question 3)
def calc_prob_cond_indep(Y, p1, p2, m1, sig1, m2, sig2):
    arr_prob_cond = p1 * scipy.stats.norm.pdf(Y, m1, sig1)
    arr_prob_cond = arr_prob_cond / (arr_prob_cond + p2 * scipy.stats.norm.pdf(Y, m2, sig2))
    return arr_prob_cond


# Partie II. Question 3)
def calc_next_prob_indep(arr_prob_cond):
    return np.mean(arr_prob_cond)


# Partie II. Question 3)
def calc_next_mu_indep(Y, arr_prob_cond):
    return np.sum((Y * arr_prob_cond)) / np.sum(arr_prob_cond)


# Partie II. Question 3)
def calc_next_sig_indep(Y, arr_prob_cond, next_mu):
    return np.sum((((Y - next_mu) ** 2) * arr_prob_cond)) / np.sum(arr_prob_cond)


# Partie II. Question 3)
def estim_param_EM_indep(K, Y, p1, p2, m1, sig1, m2, sig2):
    p1_K, p2_K, m1_K, sig1_K, m2_K, sig2_K = p1, p2, m1, sig1, m2, sig2
    for k in range(K):
        arr_prob_cond_1 = calc_prob_cond_indep(Y=Y, p1=p1_K, p2=p2_K, m1=m1_K, sig1=sig1_K, m2=m2_K, sig2=sig2_K)
        arr_prob_cond_2 = calc_prob_cond_indep(Y=Y, p1=p2_K, p2=p1_K, m1=m2_K, sig1=sig2_K, m2=m1_K, sig2=sig1_K)
        p1_K = calc_next_prob_indep(arr_prob_cond=arr_prob_cond_1)
        p2_K = 1 - p1_K
        m1_K = calc_next_mu_indep(Y=Y, arr_prob_cond=arr_prob_cond_1)
        sig1_K = np.sqrt(calc_next_sig_indep(Y=Y, arr_prob_cond=arr_prob_cond_1, next_mu=m1_K))
        m2_K = calc_next_mu_indep(Y=Y, arr_prob_cond=arr_prob_cond_2)
        sig2_K = np.sqrt(calc_next_sig_indep(Y=Y, arr_prob_cond=arr_prob_cond_2, next_mu=m2_K))
    return p1_K, p2_K, m1_K, sig1_K, m2_K, sig2_K


# Partie II. Question 4)
def calc_init_kmeans_indep(Y):
    k_means = sklearn.cluster.KMeans(n_clusters=2, random_state=0).fit(Y.reshape(-1, 1))
    p1, p2 = utils.calc_probaprio2(k_means.labels_, 0, 1)
    m1, m2 = k_means.cluster_centers_[0][0], k_means.cluster_centers_[1][0]
    sq_dist = np.min(scipy.spatial.distance.cdist(Y.reshape(-1, 1), k_means.cluster_centers_), axis=1) ** 2
    var1 = np.sum((sq_dist * (k_means.labels_ == 0))) / np.sum(k_means.labels_ == 0)
    var2 = np.sum((sq_dist * (k_means.labels_ == 1))) / np.sum(k_means.labels_ == 1)
    sig1, sig2 = np.sqrt(var1), np.sqrt(var2)
    return p1, p2, m1, sig1, m2, sig2


# Partie II. Question 4)
def show_images_indep(img_name, m1, sig1, m2, sig2, X_img, Y_img, S_img):
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


# Partie II. Question 4)
def segmentation_image_indep(img_name, m1, sig1, m2, sig2):
    print(f'-----------\n Traitement {img_name} | m1 : {m1}, sig1 : {sig1} ; m2 : {m2}, sig2 : {sig2}')

    X = load_img_to_line(img_name)
    Y = utils.bruit_gauss2(X, 0, 255, m1, sig1, m2, sig2)

    p1_0, p2_0, m1_0, sig1_0, m2_0, sig2_0 = calc_init_kmeans_indep(Y=Y)
    p1_K, p2_K, m1_K, sig1_K, m2_K, sig2_K = estim_param_EM_indep(K=25, Y=Y, p1=p1_0, p2=p2_0, m1=m1_0, sig1=sig1_0,
                                                                  m2=m2_0, sig2=sig2_0)

    S = utils.MAP_MPM2(Y=Y, cl1=0, cl2=255, p1=p1_K, p2=p2_K, m1=m1_K, sig1=sig1_K, m2=m2_K, sig2=sig2_K)
    S_inv = (S == 0) * 255

    X_img = utils.transform_line_in_img(X, int(math.sqrt(len(X))))
    Y_img = utils.transform_line_in_img(Y, int(math.sqrt(len(Y))))
    if utils.taux_erreur(X, S) > utils.taux_erreur(X, S_inv):
        S = S_inv
    S_img = utils.transform_line_in_img(S, int(math.sqrt(len(S))))

    # show_images_indep(img_name=img_name, m1=m1, sig1=sig1, m2=m2, sig2=sig2, X_img=X_img, Y_img=Y_img, S_img=S_img)

    return utils.taux_erreur(X, S)
