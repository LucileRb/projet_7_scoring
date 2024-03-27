# projet_7_scoring

# CREER ENV VIRTUEL

1) Mettre en oeuvre un outil de "scoring crédit" pour calculer la probabilié qu'un client rembourse son crédit
2) classifier la demande en crédit accordé/refusé

-> algorithme de classification en s'appuyant sur des données variées (données comportementales, données provenant d'autres institutions financières, etc...)

-> réaliser un dashboard intéractif (Dash ou Bokeh ou Streamlit)

Données : https://www.kaggle.com/c/home-credit-default-risk/data

Liste d'outils à utliser pour créer une plateforme MLOps:
● MLFlow pour la gestion “d’expériences” et leur tracking lors de la
phase d’entraînement des modèles, ainsi que la visualisation des
résultats avec MLFlow UI, pour le partager avec Chris
● MLFlow pour le stockage centralisé des modèles dans un “model
registry” et le serving
● Git, logiciel de version de code, pour suivre les modifications du
code final de l’API de prédiction de tags à déployer
● Github pour stocker et partager sur le cloud le code de l’API,
alimenté par un “push” Git et ainsi assurer une intégration continue
● Github Actions pour le déploiement continu et automatisé du code
de l’API sur le cloud
● Pytest (ou Unittest) pour concevoir les tests unitaires et les
exécuter de manière automatisée lors du build réalisé par Github
Actions