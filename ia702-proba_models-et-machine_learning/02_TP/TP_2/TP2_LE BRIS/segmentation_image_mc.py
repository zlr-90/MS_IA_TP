import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from codes.utils import (bruit_gauss2, taux_erreur,
                         transform_peano_in_img, peano_transform_img)
from scipy.stats import norm
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from multiprocess import Pool, cpu_count
from collections import deque
import seaborn as sns
from time import time

sns.set_theme(style="whitegrid")


def gauss2(Y, m1, sig1, m2, sig2):
    mat_f = np.zeros((Y.size, 2))
    mat_f[:, 0] = norm.pdf(Y, loc=m1, scale=sig1)
    mat_f[:, 1] = norm.pdf(Y, loc=m2, scale=sig2)
    
    return mat_f


def forward2(mat_f, A, p1):
    alpha = np.zeros(mat_f.shape)
    size = mat_f.shape[0]
    
    # Initialisation
    alpha[0, 0] = mat_f[0, 0] * p1
    alpha[0, 1] = mat_f[0, 1] * (1 - p1)
    alpha[0, :] = alpha[0, :] / alpha[0, :].sum()
    
    # Itération
    for i in range(1, size):
        alpha[i, :] = np.dot(alpha[i-1, :], A) * mat_f[i, :]
        alpha[i, :] = alpha[i, :] / alpha[i, :].sum()
    
    return alpha


def backward2(mat_f, A, p1):
    beta = np.zeros(mat_f.shape)
    size = mat_f.shape[0]
    
    # Initialisation
    beta[-1, 0] = 0.5
    beta[-1, 1] = 0.5
    
    # Récurrence
    for i in range(size - 2, -1, -1):
        #beta[i, :] = np.dot(beta[i+1, :], A) * mat_f[i+1, :]
        beta[i, 0] = (beta[i+1, 0] * A[0, 0] * mat_f[i+1, 0] +
                      beta[i+1, 1] * A[0, 1] * mat_f[i+1, 1])
        beta[i, 1] = (beta[i+1, 0] * A[1, 0] * mat_f[i+1, 0] +
                      beta[i+1, 1] * A[1, 1] * mat_f[i+1, 1])
        beta[i, :] = beta[i, :] / beta[i, :].sum()
    
    return beta


def MPM_chaines2(mat_f, cl1, cl2, A, p1):
    alpha = forward2(mat_f, A, p1)
    beta = backward2(mat_f, A, p1)
    
    chi = alpha * beta
    
    result = cl1 * np.ones(mat_f.shape[0])
    result[chi[:, 0] < chi[:, 1]] = cl2
    
    return np.array(result)


def calc_probaprio_mc(X, cl1, cl2):
    A = np.zeros((2, 2))
    X = 0 * (X == cl1) + 1 * (X == cl2)
    
    p1 = (X == cl1).sum() / X.shape[0]
    
    for i in range(X.shape[0] - 1):
        A[X[i], X[i + 1]] += 1
        
    return p1, A / A.sum(axis=1).reshape(-1, 1)


def estim_param_EM_mc(n_iter: float, Y, A, p1, m1, sig1, m2, sig2):
    Y = np.array(Y)
    A = np.array(A)
    size = Y.shape[0]
    param = [list(A.flatten()) + [p1, m1, sig1, m2, sig2]]
    flag = True
    cst_count = True
    
    # Si n_iter est plus grand que 1, on itère round(n_iter) fois
    # Si n_iter est plus petit que 1, on stoppe dès que toutes variations
    # relatives des paramètres d'une itération à l'autre sont < n_iter
    if n_iter >= 1:
        n_iter = int(n_iter)
        count = 0
    else:
        assert n_iter > 0
        threshold = n_iter
        cst_count = False
        
    # Début de l'itération
    while flag:
        mat_f = gauss2(Y, m1, sig1, m2, sig2)
        alpha = forward2(mat_f, A, p1)
        beta = backward2(mat_f, A, p1)
        
        # Calcul de psi selon la formule :
        # psi[n, i, j] = alpha[n, i] * A[i, j] * mat_f[n+1, j] * beta[n+1, j]
        psi = np.array((size - 1, 2, 2))
        psi = (alpha[0:size-1, :].reshape((size-1, 2, -1)) *
               A.reshape((-1, 2, 2)) * 
               mat_f[1:size, :].reshape((size-1, -1, 2)) *
               beta[1:size, :].reshape((size-1, -1, 2)))
        psi = psi / psi.sum(axis=(1, 2)).reshape(-1, 1, 1)
        
        # Calcul de chi       
        chi = alpha * beta
        chi = chi / chi.sum(axis=1).reshape(-1, 1)
        chi_sum = chi.sum(axis=0)
        
        # Calcul et enregistrement des paramètres
        p1 = chi_sum[0] / size
        A = psi.sum(axis=0) / chi_sum.reshape(-1, 1)
        m1, m2 = np.dot(Y, chi) / chi_sum
        sig1 = np.dot((Y - m1) ** 2, chi[:, 0]) / chi_sum[0]
        sig1 = np.sqrt(sig1)
        sig2 = np.dot((Y - m2) ** 2, chi[:, 1]) / chi_sum[1]
        sig2 = np.sqrt(sig2)
        param.append(list(A.flatten()) + [p1, m1, sig1, m2, sig2])
        
        # On évalue s'il faut stopper l'itération ou non selon le critère choisi
        if cst_count:
            count += 1
            if count == n_iter:
                flag = False
        else:
            evol = np.array(param[-2:][:])
            evol = np.abs((evol[1, :] - evol[0, :]) / evol[0, :])
            if evol.max() < threshold:
                flag = False
        
    return np.array(param)


class Segmentation_EM_mc():

    def __init__(self, paths, bruits, n_iter, n_jobs=3, backend='threading',
                 estimator=estim_param_EM_mc):
        self.paths = paths
        self.bruit = bruits
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.backend = backend
        self.estimator = estimator
        self.X = {}
        self.Y = {}
        self.X_segmented = {}
        self.erreur = {}
        self.param = {}
        self.duration = {}

    def process_unit(self, path, bruit):
        """
        Segmentation unitaire sur une image et un niveau de bruit donnés 
        """        
        # Acquisition et bruitage de l'image
        image = cv2.imread(path, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        X = peano_transform_img(image)
        cl1 = np.min(X)
        cl2 = np.max(X)

        m1, sig1, m2, sig2 = bruit
        Y = bruit_gauss2(X, cl1, cl2, m1, sig1, m2, sig2)
        
        # Initialisation des paramètres
        time1 = time()
        clusters = KMeans(n_clusters=2)
        clusters.fit(Y.reshape(-1, 1))
        
        group1 = Y[clusters.labels_ == 0]
        m1, sig1 = group1.mean(), group1.std()
        group2 = Y[clusters.labels_ == 1]
        m2, sig2 = group2.mean(), group2.std()
        p1, A = calc_probaprio_mc(clusters.labels_, 0, 1)

        # Estimation des paramètres
        param = self.estimator(self.n_iter, Y, A, p1, m1, sig1, m2, sig2)
        time2 = time()
        
        a11, a12, a21, a22, p1, m1, sig1, m2, sig2 = param[-1, :]
        A = np.array([[a11, a12], [a21, a22]])
        mat_f = gauss2(Y, m1, sig1, m2, sig2)

        # Segmentation de l'image
        X_segmented = MPM_chaines2(mat_f, cl1, cl2, A, p1)

        # Calcul du taux d'erreur
        erreur = taux_erreur(X, X_segmented)
        X_segmented_mirror = np.where(X_segmented == cl1, cl2, cl1)
        erreur_mirror = taux_erreur(X, X_segmented_mirror)
        if erreur_mirror < erreur:
            X_segmented = X_segmented_mirror
            erreur = erreur_mirror

        return X, Y, X_segmented, erreur, bruit, param, time2 - time1

    def process_path(self, path):
        """
        Segmentation sur une image donnée, avec différent niveaux de bruit
        (utilisation du multithreading pour accélerer le traitement) 
        """ 
        # Multithreading
        results = Parallel(n_jobs=self.n_jobs, backend=self.backend)(delayed(
           self.process_unit)(path, bruit) for bruit in self.bruit)
        X, Y, X_segmented, erreur, bruit0, param, duration = zip(*results)
        
        # Affichage d'informations
        print(f"\tImage {path[7:-4]} traitée !")
        print(f"\tItérations et temps de traitement nécessaires : ")
        for i in range(len(param)):
            print(f"\tBruit {bruit0[i]} : {len(param[i]) - 1} itérations "
                  f"et {duration[i]:.5f} seconde(s).")
        print("\n")
        
        return X, Y, X_segmented, erreur, bruit0, path, param, duration

    def process(self):
        """
        Segmentation sur l'ensemble des images, avec les différent niveaux
        de bruit (utilisation du multiprocessing pour accélerer le traitement) 
        """ 
        self.X = {}
        self.Y = {}
        self.X_segmented = {}
        self.erreur = {}
        self.bruit_path = {}
        self.param = {}
        self.duration = {}
        
        print(f'Début du traitement sur {cpu_count()} cores')
        with Pool() as pool: # Multiprocessing
            results = pool.map(self.process_path, self.paths)
        print("Fin du traitement !\n")
        
        # Enrichissement des attributs de l'objet avec les résultats générés
        Xs, Ys, Xs_segmented, erreurs, bruits0, paths, params, durations = zip(*results)
        for i, path in enumerate(paths):
            self.X[path] = Xs[i][0]
            self.Y[path] = Ys[i]
            self.X_segmented[path] = Xs_segmented[i]
            self.erreur[path] = erreurs[i]
            self.bruit_path[path] = bruits0[i]
            self.param[path] = params[i]
            self.duration[path] = durations[i]

    def export_image(self, path_image, figsize):
        """
        Export des résultats en image
        """ 
        X = self.X
        Y = self.Y
        X_segmented = self.X_segmented
        erreur = self.erreur
        n_images = len(self.paths)

        assert  n_images > 0, 'No result to display !'
        
        for i, bruit in enumerate(self.bruit):
            fig, ax = plt.subplots(nrows=n_images, ncols=3, figsize=figsize)
            plt.grid(None)

            for j, path in enumerate(self.paths):
                size = int(np.sqrt(X[path].shape[0]))
                fig.suptitle((f"Modèle des chaînes de Markov cachées (méthode EM)\n"
                              f"Bruit (m1, sig1, m2, sig2) = {bruit}\n"),
                             fontsize="large")
                
                # S'il y a seulement une image, ax n'a qu'une dimension
                if len(self.paths) > 1:
                    ax2 = ax[j, :]
                else:
                    ax2 = ax
                
                # Affichage de l'image originale
                ax2[0].imshow(transform_peano_in_img(X[path], size),
                                cmap='gray')
                ax2[0].set_title("")
                ax2[0].set_ylabel(f"{path[7: -4]}", rotation=90, size='large')
                
                # Affichage de l'image bruitée
                ax2[1].imshow(transform_peano_in_img(Y[path][i], size),
                                cmap='gray')
                ax2[1].set_title("")
                
                # Affichage de l'image segmentée
                ax2[2].imshow(transform_peano_in_img(X_segmented[path][i], size),
                                cmap='gray')
                ax2[2].set_title(f"Taux d'erreur: {erreur[path][i]:.4f}")

            fig.tight_layout()
            fig.savefig(path_image + f"{bruit}.jpg", pil_kwargs={'quality': 80})
            plt.close(fig)
            
            
    def export_variation_param(self, path_image, figsize):
        for path in self.paths:
            fig, ax = plt.subplots(nrows=1, ncols=len(self.bruit),
                                   figsize=figsize)
            fig.suptitle((f"Image : {path[7: -4]}"), fontsize="large")
            
            for i, bruit in enumerate(self.bruit):
                params = self.param[path][i]
                var_param = np.abs((params[1:, :] - params[:-1, :]) / params[:-1, :])
                var_param = pd.DataFrame(data=var_param,
                                         columns=['a11', 'a12', 'a21', 'a22', 'p1',
                                                  'm1', 'sig1', 'm2', 'sig2'])
                var_param.index = np.arange(1, len(var_param) + 1)
                sns.lineplot(data=var_param, ax=ax[i])
                ax[i].set_yscale('log')
                ax[i].set_title(f"Bruit : {bruit}")
                ax[i].set_xlabel("Itérations")
            
            fig.tight_layout()
            fig.savefig(path_image + f"{path[7: -4]}.jpg", pil_kwargs={'quality': 80})
            plt.close(fig)   
        
        
if __name__ == '__main__':

    # Question 8
    path = ['images/cible2.bmp', 'images/promenade2.bmp', 'images/city2.bmp']
    bruit = [(0, 1, 3, 2), (1, 1, 1, 5), (0, 1, 1, 1)]
    
    session = Segmentation_EM_mc(path, bruit, 1e-3)
    session.process()
    session.export_image('results/image_EMmc_1e-3_', figsize=(10, 10))
    session.export_variation_param('results/var_EMmc_1e-3_', figsize=(10, 5))