# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:25:35 2021

@author: antoi
"""
import cv2 as cv
import numpy as np
from scipy.stats import norm
import utils as u
from sklearn.cluster import KMeans 
import pandas as pd
import seaborn as sns

def Mat_f_gauss2(Y,m1,sig1,m2,sig2):
    Mat_f = np.zeros((len(Y), 2))
    Mat_f[:,0] = norm.pdf(Y, m1, sig1)
    Mat_f[:,1] = norm.pdf(Y, m2, sig2)
    return Mat_f

def forward2(Mat_f,A,p10,p20):
    n, i = Mat_f.shape
    alpha = np.zeros((n,2))
    alpha[0,0] = p10 * Mat_f[0,0]
    alpha[0,1] = p20 * Mat_f[0,1]
    alpha[0,:] = alpha[0,:] / np.sum(alpha[0,:])
    for k in range(1,n):
        alpha[k,:] = alpha[k-1,:] @ A * Mat_f[k,:]
        alpha[k,:] = alpha[k,:]/np.sum(alpha[k,:])
    print(alpha)
    return alpha

def backward2(Mat_f,A):
    n, i = Mat_f.shape
    beta = np.zeros((n,2))
    beta[0,:] = np.array([1, 1])
    beta[0,:] = beta[0,:]/np.sum(beta[0,:])
    Mat_f_flip = np.flip(Mat_f, 0)
    for k in range(1,n):
        beta[k,:] = (beta[k-1,:] * Mat_f_flip[k-1,:]) @ A
        beta[k,:] = beta[k,:]/np.sum(beta[k,:])
    beta = np.flip(beta, 0)
    return beta

def MPM_chaines2(Mat_f, cl1, cl2, A, p10, p20):
    
    alpha = forward2(Mat_f,A,p10,p20)
    beta = backward2(Mat_f,A)
    
    epsi1 = alpha[:,0] * beta[:,0]
    epsi2 = alpha[:,1] * beta[:,1]
    
    X_apost = np.where((epsi1 > epsi2),cl1,cl2)
    return X_apost

def init_param_em_mc(Y):
    #frequence de 00 01 et 10 11 
    #p10 = proportionn de chacune des classes
    
    kmeans = KMeans(n_clusters = 2, random_state = 0).fit(Y.reshape(-1,1))

    p1 = np.sum(kmeans.labels_)/len(kmeans.labels_)
    p2 = 1 - p1
    
    m1 = np.sum(np.multiply(Y, kmeans.labels_))/np.sum(kmeans.labels_)
    m2 = np.sum(np.multiply(Y, (1 - kmeans.labels_)))/np.sum((1 - kmeans.labels_))

    sig1 = np.sum(np.multiply((Y - m1)**2, kmeans.labels_)) / np.sum(kmeans.labels_)
    sig2 = np.sum(np.multiply((Y - m2)**2, (1 - kmeans.labels_)))/np.sum((1 - kmeans.labels_))
    
    dic = {}
    dic['00'] = 0
    dic['01'] = 0
    dic['11'] = 0
    dic['10'] = 0
    prev = kmeans.labels_[0]
    label1 = np.sum(kmeans.labels_)
    label0 = len(Y) - label1
    for l in kmeans.labels_[1:] :
        if l == prev :
            if prev == 0:
                dic['00'] += 1 / label0
            else :
                dic['11'] += 1 / label1
        else :
            if prev == 0:
                dic['01'] += 1 / label0
            else :
                dic['10'] += 1 / label1
        prev = l
    A = np.zeros((2,2))
    A[0,0] = dic['00']
    A[0,1] = dic['01']
    A[1,0] = dic['10']
    A[1,1] = dic['11']
    
    return A, p1, p2, m1, sig1, m2, sig2

def estim_param_EM_mc(itera, Y, A, p10, p20, m1, sig1, m2, sig2):
    
    n = len(Y)
    Aem, p1em, p2em, m1em, sig1em, m2em, sig2em = A, p10, p20, m1, sig1, m2, sig2

    for k in range(itera):

        #mettre à jour alpha et beta Mat_f
        #psi n*2
        #ksi n-1*2*2
        
        Mat_f = Mat_f_gauss2(Y,m1em,sig1em,m2em,sig2em)
        alpha = forward2(Mat_f,Aem,p1em,p2em)
        beta = backward2(Mat_f,Aem)
        
        psi1 = alpha[:-1,0] * Aem[0,0] * Mat_f[1:,0] * beta[1:,0]
        psi2 = alpha[:-1,0] * Aem[0,1] * Mat_f[1:,1] * beta[1:,1]
        psi3 = alpha[:-1,1] * Aem[1,0] * Mat_f[1:,0] * beta[1:,0]
        psi4 = alpha[:-1,1] * Aem[1,1] * Mat_f[1:,1] * beta[1:,1]
        summ = psi1 + psi2 + psi3 + psi4
        psi1 = psi1 / summ
        psi2 = psi2 / summ
        psi3 = psi3 / summ
        psi4 = psi4 / summ
        
        epsi1 = alpha[:,0] * beta[:,0]
        epsi2 = alpha[:,1] * beta[:,1]
        summ2 = epsi1 + epsi2
        epsi1 = epsi1 / summ2
        epsi2 = epsi2 / summ2
        
        p1em = np.sum(epsi1) / n
        p2em = np.sum(epsi2) / n
        
        m1em = np.sum(Y * epsi1) / np.sum(epsi1)
        m2em = np.sum(Y * epsi2) / np.sum(epsi2)
        
        sig1em = np.sum((Y - m1em)**2 * epsi1) / np.sum(epsi1)
        sig2em = np.sum((Y - m2em)**2 * epsi2) / np.sum(epsi2)
        
        Aem[0,0] = np.sum(psi1) / np.sum(epsi1[-1])
        Aem[0,1] = np.sum(psi2) / np.sum(epsi1[-1])
        Aem[1,0] = np.sum(psi3) / np.sum(epsi2[-1])
        Aem[1,1] = np.sum(psi4) / np.sum(epsi2[-1])

    return Aem, p1em, p2em, m1em, sig1em, m2em, sig2em

def Segmentation_image_mc(path, cl1, cl2, m1i, sig1i, m2i, sig2i, itera):
    
    cvImage = cv.cvtColor(cv.imread(path),cv.COLOR_BGR2GRAY)

    image1d = u.peano_transform_img(cvImage)
    
    Y = u.bruit_gauss2(image1d, cl1, cl2, m1i, sig1i, m2i, sig2i)
    
    A, p1, p2, m1, sig1, m2, sig2 = init_param_em_mc(Y)

    Yplot = u.transform_peano_in_img(Y, 256)

    A, p1em, p2em, m1em, sig1em, m2em, sig2em = estim_param_EM_mc(itera, Y, A, p1, p2, m1, sig1, m2, sig2)
    
    Mat_f = Mat_f_gauss2(Y, m1em, sig1em, m2em, sig2em)
    segmentation = MPM_chaines2(Mat_f, cl1, cl2, A, p1em, p2em)
    segmentation_plot = u.transform_peano_in_img(segmentation, 256)
    
    return u.taux_erreur(segmentation, image1d), cvImage, Yplot, segmentation_plot

erreur, cvImage, Yplot, segmentation_plot = Segmentation_image_mc('C:/Users/antoi/Documents/Telecom Paris/Cours/IA702 Probabilistic model/TP3/beee2.bmp',255,0,0,1,3,1,5)