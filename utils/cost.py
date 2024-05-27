########## Imports ##########
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from utils import fonctions
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler


########## Fonction de coût ##########

# 1ère fonction coût avec modification du f1score 
#def custom_metric(y_true, y_pred):
#    '''function pour pénaliser faux positifs en particulier'''
#    tp, tn, fp, fn = confusion_matrix(y_true, y_pred).ravel()
#    tp_weight = 1
#    tn_weight = 1
#    fn_weight = -1
#    fp_weight = -10
#    tp = tp * tp_weight
#    tn = tn * tn_weight
#    fn = fn * fn_weight
#    fp = fp * fp_weight
#    recall = tp / (tp + fn)
#    precision = tp / (tp + fp)
#    f1score = 2 * (precision * recall) / (precision + recall)
#    return f1score

# -> intéressant mais résultats ne sautent pas aux yeux, besoin de les interpréter
# -> construire une fonction coût plus simple, sans passer par le f1score


def custom_metric(y_true, y_pred):
    '''
    Fonction pour pénaliser faux négatifs (prêt accordé non remboursé) en particulier
    -------------------------------------------------------------------------------
    Matrice de confusion puis on fait la somme des tp, tn, fn, fp après pondération
    Résultat:
    - quand la somme est positive : on fait de l'argent
    - quand elle est négative : on en perd
    '''

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Attribuer poids en fonction du résultat de la matrice de confusion
    tp_weight = 1 # vrai positif - prêt accordé et remboursé
    tn_weight = 1 # vrai négatif - prêt non accordé et non remboursé
    fp_weight = -1 # faux positif - prêt non accordé qui aurait été remboursé -> refus crédit et manque à gagner en marge
    fn_weight = -10 # faux négatif - prêt accordé et non remboursé -> crédit accordé et perte en capital

    tp_w = tp * tp_weight
    tn_w = tn * tn_weight
    fn_w = fn * fn_weight
    fp_w = fp * fp_weight

    score = tp_w + tn_w + fn_w + fp_w

    return score

############################### END ###############################z