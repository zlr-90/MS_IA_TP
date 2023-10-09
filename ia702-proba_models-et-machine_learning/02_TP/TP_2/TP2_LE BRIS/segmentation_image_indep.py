import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from codes.utils import line_transform_img, bruit_gauss2, transform_line_in_img, \
    MAP_MPM2, taux_erreur
from scipy.stats import norm
from sklearn.cluster import KMeans
from multiprocess import Pool, cpu_count
from joblib import Parallel, delayed
from collections import deque
from time import time
import seaborn as sns

sns.set_theme(style="whitegrid")


# Question 3
def estim_param_EM_indep(n_iter: float, Y, p1, m1, sig1, m2, sig2):
    Y = np.array(Y)
    size = Y.shape[0]
    params = [[p1, m1, sig1, m2, sig2]]
    flag = True
    cst_count = True
    
    if n_iter >= 1:
        n_iter = int(n_iter)
        count = 0
    else:
        assert n_iter > 0
        threshold = n_iter
        cst_count = False

    while flag:
        temp1 = p1 * norm.pdf(Y, loc=m1, scale=sig1)
        temp2 = (1 - p1) * norm.pdf(Y, loc=m2, scale=sig2)

        temp12_sum = temp1 + temp2
        posterior1 = temp1 / temp12_sum
        posterior2 = temp2 / temp12_sum

        posterior1_sum = posterior1.sum()
        posterior2_sum = posterior2.sum()

        p1 = posterior1_sum / size
        m1 = np.dot(Y, posterior1) / posterior1_sum
        m2 = np.dot(Y, posterior2) / posterior2_sum
        sig1 = np.dot((Y - m1) ** 2, posterior1) / posterior1_sum
        sig1 = np.sqrt(sig1)
        sig2 = np.dot((Y - m2) ** 2, posterior2) / posterior2_sum
        sig2 = np.sqrt(sig2)
        params.append([p1, m1, sig1, m2, sig2])
        
        if cst_count:
            count += 1
            if count == n_iter:
                flag = False
        else:
            evol = np.array(params[-2:][:])
            evol = np.abs((evol[1, :] - evol[0, :]) / evol[0, :])
            if evol.max() < threshold:
                flag = False
                
    return np.array(params)


# Implémentation d'une classe pour simplifier la production de résultats
class Segmentation_inde():

    def __init__(self, paths, bruits, n_iter, n_jobs=3, backend='threading'):
        self.paths = paths
        self.bruit = bruits
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.backend = backend
        self.X = {}
        self.Y = {}
        self.X_segmented = {}
        self.erreur = {}
        self.param = {}
        self.duration = {}

    def process_unit(self, path, bruit):
        """
        Segmentation unitaire sur une image et un niveau de bruit 
        """        
        # Acquisition et bruitage de l'image
        image = cv2.imread(path, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        X = line_transform_img(image)
        cl1 = np.min(X)
        cl2 = np.max(X)

        m1, sig1, m2, sig2 = bruit
        Y = bruit_gauss2(X, cl1, cl2, m1, sig1, m2, sig2)
        time1 = time()

        # Initialisation des paramètres
        clusters = KMeans(n_clusters=2)
        clusters.fit(Y.reshape(-1, 1))
        group1 = Y[clusters.labels_ == 0]
        group2 = Y[clusters.labels_ == 1]
        p1 = group1.size / (group1.size + group2.size)
        m1 = group1.mean()
        m2 = group2.mean()
        sig1 = group1.std()
        sig2 = group2.std()

        # Estimation des paramètres
        param = estim_param_EM_indep(self.n_iter, Y, p1, m1, sig1, m2, sig2)
        p1, m1, sig1, m2, sig2 = param[-1, :]
        time2 = time()

        # Segmentation de l'image
        X_segmented = MAP_MPM2(Y, cl1, cl2, p1, 1-p1, m1, sig1, m2, sig2)

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
        Segmentation sur une image, avec différent niveaux de bruit
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
        
        # Enrichissement des attributs de l'objet avec les résultats
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
            fig, ax = plt.subplots(nrows=n_images,
                                   ncols=3, figsize=figsize)
            plt.grid(None)

            for j, path in enumerate(self.paths):
                size = int(np.sqrt(X[path].shape[0]))
                fig.suptitle((f"Modèle des couples indépendants\n"
                              f"Bruit (m1, sig1, m2, sig2) = {bruit}\n"),
                             fontsize="large")
                
                # S'il y a seulement une image, ax n'a qu'une dimension
                if len(self.paths) > 1:
                    ax2 = ax[j, :]
                else:
                    ax2 = ax
                
                # Affichage de l'image originale
                ax2[0].imshow(transform_line_in_img(X[path], size),
                                cmap='gray')
                ax2[0].set_title("")
                ax2[0].set_ylabel(f"{path[7: -4]}", rotation=90, size='large')
                
                # Affichage de l'image bruitée
                ax2[1].imshow(transform_line_in_img(Y[path][i], size),
                                cmap='gray')
                ax2[1].set_title("")
                
                # Affichage de l'image segmentée
                ax2[2].imshow(transform_line_in_img(X_segmented[path][i], size),
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
                                         columns=['p1', 'm1', 'sig1', 'm2', 'sig2'])
                var_param.index = np.arange(1, len(var_param) + 1)
                sns.lineplot(data=var_param, ax=ax[i])
                ax[i].set_yscale('log')
                ax[i].set_title(f"Bruit : {bruit}")
                ax[i].set_xlabel("Itérations")
            
            fig.tight_layout()
            fig.savefig(path_image + f"{path[7: -4]}.jpg", pil_kwargs={'quality': 80})
            plt.close(fig) 


if __name__ == '__main__':

    # Question 4
    path = ['images/zebre2.bmp']
    bruit = [(1, 1, 1, 5)]
    session = Segmentation_inde(path, bruit, 200)
    session.process()
    session.export_image('results/test_inde_', figsize=(10, 4))

    # Question 5
    path = ['images/cible2.bmp', 'images/promenade2.bmp', 'images/city2.bmp']
    bruit = [(0, 1, 3, 2), (1, 1, 1, 5), (0, 1, 1, 1)]
    
    session = Segmentation_inde(path, bruit, 50)
    session.process()
    session.export_image('results/image_inde_5O_', figsize=(10, 10))
    session.export_variation_param('results/var_inde_50_', figsize=(10, 5))
    
    session = Segmentation_inde(path, bruit, 1e-3)
    session.process()
    session.export_image('results/image_inde_1e-3_', figsize=(10, 10))
    
    
    