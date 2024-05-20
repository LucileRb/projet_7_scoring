# Implement a scoring model

# Descriptif du projet
1) Mettre en oeuvre un outil de "scoring crédit" pour calculer la probabilié qu'un client rembourse son crédit
2) classifier la demande en crédit accordé/refusé

-> algorithme de classification en s'appuyant sur des données variées (données comportementales, données provenant d'autres institutions financières, etc...)
-> réaliser un dashboard intéractif (Dash ou Bokeh ou Streamlit)

Données : https://www.kaggle.com/c/home-credit-default-risk/data


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

f beta score -> à regarder

make_scorer -> à regarder - à utiliser aussi pour gridsearch, etc...

# Evaluation des modèles
gridsearchCV
comparer/créer modèles -> baseline = accorder de façon aléatoire des crédits (genre tirer à pile ou face si on accorde les crédits ou non)

# Analyse de l'importance des features (globales ou locales)
craft AI -> plateforme de MLOps
mise en prod

Modélisation : algos Catboost, LightGBM & XGBoost

# se focaliser sur une seule métrique pour prendre des décisions
# quand plusieurs métriques -> on ne sait plus laquelle suivre

Setup Mac bureau : python 3.11.3 | MacBook Air puce Apple M2 OS Sonoma version 14.4.1

Inspirations
https://github.com/eleplanois/openclassRoom/tree/main/Projet_7%20Impl%C3%A9mentez%20un%20mod%C3%A8le%20de%20scoring
https://github.com/nalron/project_credit_scoring_model


HEROKU
Procfile = fichier texte à la racine du dossier de l'application, pour déclarer quelle commande executer pour démarrer l'app
https://devcenter.heroku.com/articles/getting-started-with-python#create-and-deploy-the-app
https://devcenter.heroku.com/articles/heroku-cli-commands#heroku-apps-create-app
https://lit-cove-87268-9b9a2d0fbdb8.herokuapp.com/ | https://git.heroku.com/lit-cove-87268.git
When you create an app, a git remote called heroku is also created and associated with your local git repository. Git remotes are versions of your repository that live on other servers. You deploy your app by pushing its code to that special Heroku-hosted remote associated with your app.

Deploy your code. This command pushes the main branch of the sample repo to your heroku remote, which then deploys to Heroku

modif