########## Imports ##########

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from IPython.display import display

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


########## Catégorie ##########
def get_category(row):
    match = re.search(r'(?:\w+\s+){2}(?=\>\>)', row['product_category_tree'])
    if match:
        result = match.group(0).strip()
        return result

################################ END ########################################