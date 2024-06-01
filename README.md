# Projet de Scoring Crédit
Ce projet vise à développer un modèle de scoring de crédit pour prédire la probabilité de remboursement d'un prêt par un client. Le modèle est destiné à être utilisé par les chargés de relation client d'une institution financière pour évaluer les demandes de crédit et prendre des décisions éclairées sur l'octroi de prêts.

*Lien vers les données : https://www.kaggle.com/c/home-credit-default-risk/data*

### Structure du Projet
Le projet est organisé via les dossiers suivants :
- **scoring_app** : Répertoire qui contient l'ensemble des codes du dashboard intéractif (API et test unitaire, interface intéractive...)
- **notebooks** : Contient l'ensemble des notebooks au format ipynb dont notamment le fichier "eda.ipynb" (notebook pour l'exploration et le preprocessing des données) et le fichier "modelisations.ipynb" (notebook pour l'entraînement et l'évaluation des modèles)
- **utils** : Contient les scripts et les modules Python utilisés pour le prétraitement des données, l'entraînement du modèle, l'évaluation et l'interprétation des résultats, ainsi que les fichiers sauvegardés au format pickle (modèle entraîné, scaler...) et un échantillon réduit des données au format parquet
- **results** : Contient les résultats de l'analyse de data drift

Fichiers ignorés (.gitignore)
- **data** : Ce répertoire contient les données utilisées pour l'entraînement et le test du modèle (dossier ignoré par github)
- **mlruns** : Répertoire utilisé par MLflow pour enregistrer les résultats de l'entraînement des modèles et les métriques associées.
- **mlartifacts** : Contient les artefacts produits lors de l'entraînement des modèles, tels que les modèles entraînés et les visualisations.
- **cours_et_exemples** : Contient des ressources pédagogiques et des exemples utilisés pour développer le modèle (dossier ignoré par github)
- **images_presentation** : Images utilisées pour la présentation du projet.


### Dashboard intéractif

Le dashboard intéractif a été deployé sur Streamlit.io et l'API a été déployée sur Heroku.

*Lien vers l'API : https://scoring-credit-implementation-a56784ea5721.herokuapp.com/*
*Lien vers le dashboard intéractif : https://scoring-model-implementation-kicwsbv9z9kc8ysf3yjdn8.streamlit.app/*

L'interface est organisée en 3 pages:
- une page d'accueil
- une page permettant de visualiser les données d'un client, de prédire la probabilité de remboursement de son prêt et d'expliquer le résultat
- une page permettant d'entrer les données d'un nouveau client et de prédire la probabilité de remboursement de son prêt et d'expliquer le résultat

## Ressources intéressantes consultées pour ce projet
#### ML Flow
- https://mlflow.org/docs/latest/tracking.html
- https://github.com/mlflow/mlflow

#### Heroku
1) s'inscrire au programme Github (GitHub Student Developer Pack - https://education.github.com/pack)
2) inscription Heroku Student pack : https://www.heroku.com/github-students

Liste des fichiers pour déploiement heroku:
- **runtime.txt** : pour spécifier version de python utilisée dans heroku
- **requirements.txt** : pour spécifier librairies à installer et leurs versions
- **Procfile** : pour spécifier quelle app lancer et comment
fichier texte à la racine du dossier de l'application, pour déclarer quelle commande executer pour démarrer l'app
ne pas mettre d'extension après Procfile ("Procfile" et non "Procfile.txt") -> sinon le fichier ne sera pas reconnu par Heroku
 bien mettre Procfile à la racine du projet (et non dans le dossier de l'app)

https://devcenter.heroku.com/articles/getting-started-with-python#create-and-deploy-the-app
https://devcenter.heroku.com/articles/heroku-cli-commands#heroku-apps-create-app