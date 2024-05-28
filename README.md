### Projet de Scoring Crédit
Ce projet vise à développer un modèle de scoring de crédit pour prédire la probabilité de remboursement d'un prêt par un client. Le modèle est destiné à être utilisé par une institution financière pour évaluer les demandes de crédit et prendre des décisions éclairées sur l'octroi de prêts.

Lien vers les données : https://www.kaggle.com/c/home-credit-default-risk/data

# Structure du Projet
Le projet est organisé de la manière suivante :
- data/ : Ce répertoire contient les données utilisées pour l'entraînement et le test du modèle (dossier ignoré par github)
- packages/ : Contient les scripts et les modules Python utilisés pour le prétraitement des données, l'entraînement du modèle, l'évaluation et l'interprétation des résultats.
- mlruns/ : Répertoire utilisé par MLflow pour enregistrer les résultats de l'entraînement des modèles et les métriques associées.
- mlartifacts/ : Contient les artefacts produits lors de l'entraînement des modèles, tels que les modèles entraînés et les visualisations.
- scoring_app/ : Une application web simple pour démontrer l'utilisation du modèle de scoring de crédit.
- cours_et_exemples/ : Contient des ressources pédagogiques et des exemples utilisés pour développer le modèle (dossier ignoré par github)
- images_presentation/ : Images utilisées pour la présentation du projet.

Les principaux fichiers sont :
- eda.ipynb : Notebook Jupyter pour l'exploration des données.
- modelisations.ipynb : Notebook Jupyter pour l'entraînement et l'évaluation des modèles.
- data_drift.ipynb : Notebook Jupyter pour l'analyse du data drift.
- P7 - Note méthologique.pdf : Note méthodologique traitant du projet

# Dashboard
Lien vers le dashboard intéractif : https://lit-cove-87268-9b9a2d0fbdb8.herokuapp.com/

## Ressources intéressantes
# ML Flow
-> https://mlflow.org/docs/latest/tracking.html
-> https://github.com/mlflow/mlflow


# Heroku
Inscription Heroku Student pack :
https://www.heroku.com/github-students

- D'abord s'inscrire au programme Github (GitHub Student Developer Pack - https://education.github.com/pack)
-> laborieux car impossible de trouver un school ID d'openclassrooms -> https://education.github.com/discount_requests/14236253/additional_information

Procfile = fichier texte à la racine du dossier de l'application, pour déclarer quelle commande executer pour démarrer l'app
https://devcenter.heroku.com/articles/getting-started-with-python#create-and-deploy-the-app
https://devcenter.heroku.com/articles/heroku-cli-commands#heroku-apps-create-app
https://lit-cove-87268-9b9a2d0fbdb8.herokuapp.com/ | https://git.heroku.com/lit-cove-87268.git
When you create an app, a git remote called heroku is also created and associated with your local git repository. Git remotes are versions of your repository that live on other servers. You deploy your app by pushing its code to that special Heroku-hosted remote associated with your app.

Deploy your code. This command pushes the main branch of the sample repo to your heroku remote, which then deploys to Heroku

ATTENTION -> ne pas mettre d'extension après Procfile ("Procfile" et non "Procfile.txt") -> sinon le fichier ne sera pas reconnu par Heroku
ATTENTION (bis) -> bien mettre Procfile à la racine du projet (et non dans le dossier de l'app)

Liste des fichiers pour déploiement heroku:
- runtime.txt -> pour spécifier version de python utilisée dans heroku
- requirements.txt -> pour spécifier librairies à installer et leurs versions
- Procfile -> pour spécifier quelle app lancer et comment




web: sh scoring_app/setup.sh && streamlit run scoring_app/app.py
api: python scoring_app/api.py