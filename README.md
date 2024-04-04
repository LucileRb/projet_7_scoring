# OC7 : Implement a scoring model

# Environnement virtuel
python3 -m venv venv
c'est tout ????? meh
problème -> si env virtuel déjà présent -> va charger celui là au lieu d'en créer un nouveau

# Descriptif du projet
1) Mettre en oeuvre un outil de "scoring crédit" pour calculer la probabilié qu'un client rembourse son crédit
2) classifier la demande en crédit accordé/refusé

-> algorithme de classification en s'appuyant sur des données variées (données comportementales, données provenant d'autres institutions financières, etc...)
-> réaliser un dashboard intéractif (Dash ou Bokeh ou Streamlit)

Données : https://www.kaggle.com/c/home-credit-default-risk/data

Liste d'outils à utliser pour créer une plateforme MLOps:
- MLFlow pour la gestion “d’expériences” et leur tracking lors de la phase d’entraînement des modèles, ainsi que la visualisation des résultats avec MLFlow UI, pour le partager avec Chris
- MLFlow pour le stockage centralisé des modèles dans un “model registry” et le serving
- Git, logiciel de version de code, pour suivre les modifications du code final de l’API de prédiction de tags à déployer
- Github pour stocker et partager sur le cloud le code de l’API, alimenté par un “push” Git et ainsi assurer une intégration continue
- Github Actions pour le déploiement continu et automatisé du code de l’API sur le cloud
- Pytest (ou Unittest) pour concevoir les tests unitaires et les exécuter de manière automatisée lors du build réalisé par Github Actions


# ML Flow
-> https://mlflow.org/docs/latest/tracking.html
-> https://github.com/mlflow/mlflow

# Heroku
Inscription Heroku Student pack :
https://www.heroku.com/github-students

- D'abord s'inscrire au programme Github (GitHub Student Developer Pack - https://education.github.com/pack)
-> très chiant car impossible de trouver un school ID d'openclassrooms -> https://education.github.com/discount_requests/14236253/additional_information
demande faite le 01/04/24 - en attente de validation

# Feature engineering
pas besoin de pousser
2-3 variables déjà bien

# Score
personne qui ne va pas rembourser le crédit coutera 10x plus cher (pénaliser de 10)
cas contraire -> pénalisé de 1

f-score -> se rapproche pas mal de ça mais ne répond pas totalement à mon besoin
à mentionner mais créer une fonction de scoring vraiment custom

make_scorer -> à regarder - à utiliser aussi pour gridsearch, etc...

# Evaluation des modèles
gridsearchCV
comparer/créer modèles -> baseline = accorder de façon aléatoire des crédits (genre tirer à pile ou face si on accorde les crédits ou non)

# Analyse de l'importance des features (globales ou locales)




craft AI -> plateforme de MLOps
mise en prod