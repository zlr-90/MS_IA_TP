import Segmentation_image_indep as seg_img_indep
import Segmentation_image_mc as seg_img_mc
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


list_cible_val = ['alfa2.bmp', 'promenade2.bmp', 'zebre2.bmp']
list_m_val = [(0, 3), (1, 1), (0, 1)]
list_sig_val = [(1, 2), (1, 5), (1, 1)]


# Liée à aucunes questions. Juste représentation des gaussiennes
def show_all_gaussian(m_list, sig_list, nb_param):
    print('-------------\n The function show_all_gaussian is running \n-------------')
    plt.figure(figsize=(10, 7))
    for i in range(nb_param):
        plt.subplot(1, 3, i + 1)
        x1 = np.linspace(m_list[i][0] - 3 * sig_list[i][0], m_list[i][0] + 3 * sig_list[i][0], 100)
        x2 = np.linspace(m_list[i][1] - 3 * sig_list[i][1], m_list[i][1] + 3 * sig_list[i][1], 100)
        plt.plot(x1, scipy.stats.norm.pdf(x1, m_list[i][0], sig_list[i][0]))
        plt.plot(x2, scipy.stats.norm.pdf(x2, m_list[i][1], sig_list[i][1]))
        plt.title(f'Valeurs {i}')
    plt.show()


# Partie II. Question 5)
def load_sed_img_indep(list_cible, list_m, list_sig, nb_param):
    dict_error = {}
    for cible in list_cible:
        for i in range(nb_param):
            error_rate = seg_img_indep.segmentation_image_indep(img_name=cible, m1=list_m[i][0], sig1=list_sig[i][0],
                                                                m2=list_m[i][1], sig2=list_sig[i][1])
            dict_error[cible + '_' + str(i)] = error_rate
    return dict_error


# Partie III. Question 8)
def load_sed_img_mc(list_cible, list_m, list_sig, nb_param):
    dict_error = {}
    for cible in list_cible:
        for i in range(nb_param):
            error_rate = seg_img_mc.segmentation_image_mc(img_name=cible, m1=list_m[i][0], sig1=list_sig[i][0],
                                                          m2=list_m[i][1], sig2=list_sig[i][1])
            dict_error[cible + '_' + str(i)] = error_rate
    return dict_error


show_all_gaussian(m_list=list_m_val, sig_list=list_sig_val, nb_param=3)
dict_error_indep = load_sed_img_indep(list_cible=list_cible_val, list_m=list_m_val, list_sig=list_sig_val, nb_param=3)
dict_error_mc = load_sed_img_mc(list_cible=list_cible_val, list_m=list_m_val, list_sig=list_sig_val, nb_param=3)

for cible in list_cible_val:
    for i in range(3):
        print(f'Taux d\'erreur : {cible} | Méthode indep : {dict_error_indep[cible + "_" + str(i)]} ; ',
              f'Méthode mc : {dict_error_mc[cible + "_" + str(i)]}')
