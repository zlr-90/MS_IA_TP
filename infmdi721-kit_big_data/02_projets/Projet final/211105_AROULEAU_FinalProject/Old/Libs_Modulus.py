'''LISTE DES LIBRAIRIES, PACKAGES, ET MODULES'''

# Pour rendre les fichiers Excels exploitables après téléchargement
import re                                # Module d'opérations à base d'expressions rationnelles
import shutil                            # Module High-level file operations
import tempfile                          # Module de génération de fichiers et répertoires temporaires
from fnmatch      import fnmatch         # Librairie de filtrage par motif des noms de fichiers UNIX
from os           import listdir         # Pour lister les fichiers et sous-dossiers existants dans un répertoire
from os.path      import isfile, join    # Manipulation des noms de répertoires communs
from zipfile      import ZipFile         # Manipulation des fichiers ZIP

# Pour pré-traiter, traiter, et post-traiter les données
import cartopy as ctp                               # Pour manipuler les données géographiques
import datetime as dt                               # Pour manipuler les données temporelles
import dateparser
import matplotlib.pyplot as plt                     # Pour représenter les données
import numpy as np                                  # Pour manipuler les données
import pandas as pd                                 # Pour manipuler les données en forme de DataFrame
import requests                                     # Pour récupérer les données HTTP

from bs4 import BeautifulSoup                       # Pour exploiter les données d'un site internet à base de code HTML
from sklearn.metrics import r2_score                # 
from sklearn import preprocessing                   #
from sklearn.linear_model import LassoCV            #
from sklearn.linear_model import LinearRegression   # Pour études de régression linéaire

# Pour analyser et optimiser le temps de traitement
import time
import threading
from tqdm.auto import tqdm





'''Note : import et from classé dans l'ordre alphabétique pour faciliter la lecture '''