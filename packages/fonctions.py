########## Imports ##########

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from IPython.display import display

import pickle
import os

from sklearn.model_selection import train_test_split

#Preprocessing, Upsampling, Model Selection, Model Evaluation
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score 
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_predict, cross_val_score, learning_curve, cross_validate
from sklearn.feature_selection import RFECV

##########################################################################################################################################################################
########## Fonctions 'générlistes' ##########

# Labels sur graphs
def addlabels(x, y):
    """ Fonction pour ajouter valeurs sur graphs """
    for i in range(len(x)):
        plt.text(i, y[i]//2, y[i], ha = 'center', fontstyle = 'italic')


# Duplicats
def remove_duplicates(df):
    """Fonction pour détecter les doublons dans un jeu de données et les supprimer si il y en a"""

    print('********** Détection des doublons **********\n')
    
    # Nombre de duplicats dans le jeu de données
    doublons = df.duplicated().sum()
    print(f'Nombre de duplicats dans le jeu de données = {doublons}')

    if doublons > 0:

        # Affichier le pourcentage de duplicats
        print(f"\nPourcentage de duplicats : {round((df.duplicated().sum().sum()/np.product(df.shape))*100, 2)}\n")

        # supprimer duplicats
        print('****** Suppression des duplicats en cours ******')
        df.drop_duplicates(inplace = True)

        # nombre de duplicats dans le jeu de données après processing
        print(f'Nombre de duplicats dans le jeu de données après processing: {df.duplicated().sum()}')


# Données manquantes
def nan_detection(df):
    """Fonction pour détecter les données manquantes dans un jeu de données et afficher insights pertinents"""

    print('********** Détection des données manquantes **********\n')

    # Nombre total de nan :
    total_nan = df.isna().sum().sum()
    print(f'Nombre de données manquantes dans le jeu de données = {total_nan}')

    if total_nan > 0:
    
        # Pourcentage
        print(f"\nPourcentage de valeurs manquantes : {round((df.isna().sum().sum()/np.product(df.shape))*100, 2)}\n")

        # Nan et pourcentage de nan par features
        print('\nValeurs manquantes par colonne : \n')
        pd.set_option('display.max_rows', None) # pour afficher toutes les lignes
        values = df.isnull().sum()
        percentage = 100 * values / len(df)
        table = pd.concat([values, percentage.round(2)], axis = 1)
        table.columns = ['Nombres de valeurs manquantes', '% de valeurs manquantes']
        display(table[table['Nombres de valeurs manquantes'] != 0].sort_values('% de valeurs manquantes', ascending = False))
        pd.reset_option('display.max_rows') # on reset l'option pour ne plus afficher toutes les lignes

        # Heatmap
        print('\nHeatmap des valeurs manquantes : \n')
        plt.figure(figsize = (15, 7))
        sns.heatmap(df.isna(), cbar = False)
        plt.show()


def csv_describe(folder):
    '''
    Fonction pour décrire les csv contenus dans un dossier donné sous forme de dataframe:
    Shape, Valeurs manquantes, info...'''

    data_dict = {}

    for file in folder:
        data = pd.read_csv(file, encoding = 'ISO-8859-1')

        data_dict[file] = [
            data.shape[0], # Nb de lignes
            data.shape[1], # Nb de colonnes
            round(data.isna().sum().sum()/data.size*100, 2), # % valeurs manquantes
            round(data.duplicated().sum().sum()/data.size*100, 2), # % duplicats
            data.select_dtypes(include = ['object']).shape[1], # nb d'objets
            data.select_dtypes(include = ['float']).shape[1], # nb de float
            data.select_dtypes(include = ['int']).shape[1], # nb d'int
            data.select_dtypes(include = ['bool']).shape[1], # nb de bool
            round(data.memory_usage().sum()/1024**2, 3) # mémoire utilisée
            ]

        comparative_table = pd.DataFrame.from_dict(
            data = data_dict,
            columns = [
                'Rows',
                'Columns',
                '%NaN',
                '%Duplicate',
                'object_dtype',
                'float_dtype',
                'int_dtype',
                'bool_dtype',
                'MB_Memory'
                ],
            orient = 'index')
    return comparative_table


#########################################################################################
### Fonctions pour faciliter l'analyse des principales variables
# (trouvées dans divers notebooks en ligne)

def plot_stat(data, feature, title):
    """ Fonctions pour faire blablabla """

    ax, fig = plt.subplots(figsize = (7, 5))
    ax = sns.countplot(y = feature, data = data, order=data[feature].value_counts(ascending = False).index)
    ax.set_title(title)

    for p in ax.patches:
                percentage = '{:.1f}%'.format(100 * p.get_width()/len(data[feature]))
                x = p.get_x() + p.get_width()
                y = p.get_y() + p.get_height()/2
                ax.annotate(percentage, (x, y), fontsize = 14, fontweight = 'bold')

    plt.show()


def plot_percent_target1(data, feature, title):

    """ Fonctions pour faire blablabla """

    cat_perc = data[[feature, 'TARGET']].groupby([feature], as_index = False).mean()
    cat_perc.sort_values(by = 'TARGET', ascending = False, inplace = True)
    
    ax, fig = plt.subplots(figsize = (7, 5))
    ax = sns.barplot(y = feature, x = 'TARGET', data = cat_perc)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('Percent of target with value 1')

    for p in ax.patches:
                percentage = '{:.1f}%'.format(100 * p.get_width())
                x = p.get_x() + p.get_width()
                y = p.get_y() + p.get_height()/2
                ax.annotate(percentage, (x, y), fontsize = 14, fontweight = 'bold')
    plt.show()



#Plot distribution of one feature
def plot_distribution(df, feature, title):
    plt.figure(figsize = (7, 5))

    t0 = df.loc[df['TARGET'] == 0]
    t1 = df.loc[df['TARGET'] == 1]

    sns.kdeplot(t0[feature].dropna(), color = 'blue', label = 'TARGET = 0')
    sns.kdeplot(t1[feature].dropna(), color = 'red', label = 'TARGET = 1')
    plt.title(title)
    plt.ylabel('')
    plt.legend()
    plt.show()



def cf_matrix_roc_auc(model, y_true, y_pred, y_pred_proba, roc_auc):
    '''This function will make a pretty plot of 
  an sklearn Confusion Matrix using a Seaborn heatmap visualization + ROC Curve.'''
    fig = plt.figure(figsize = (20, 15))
  
    plt.subplot(221)
    cf_matrix = confusion_matrix(y_true, y_pred)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
  
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot = labels, fmt = '', cmap = 'Blues')

    plt.subplot(222)
    fpr,tpr,_ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, color = 'orange', linewidth = 5, label = 'AUC = %0.4f' %roc_auc)
    plt.plot([0, 1], [0, 1], color = 'darkblue', linestyle = '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    plt.show()

#########################################################################################


################################ END ########################################