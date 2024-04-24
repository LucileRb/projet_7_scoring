########## Imports ##########
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import fonctions
from sklearn.metrics import confusion_matrix


########## Fonction de coût ##########


def custom_metric(y_true, y_pred):
    '''function pour pénaliser faux positifs en particulier'''

    tp, tn, fp, fn = confusion_matrix(y_true, y_pred).ravel()

    tp_weight = 1
    tn_weight = 1
    fn_weight = -1
    fp_weight = -10

    tp = tp * tp_weight
    tn = tn * tn_weight
    fn = fn * fn_weight
    fp = fp * fp_weight

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1score = 2 * (precision * recall) / (precision + recall)

    return f1score

# construire une fonction cout sans passer par le f1score
# faire la somme des tp tn fn fp
# quand la somme est positive : on fait de l'argent
# quand elle est négative : on en perd