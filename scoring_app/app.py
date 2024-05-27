#### IMPORTS #####
import base64
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt
import requests

def get_prediction(data):
    api_url = 'http://18.201.108.103:8501/Prediction' # à remplacer par bonne url de l'api streamlit
    print(data)
    response = requests.post(api_url, json = data)
    print('réponse api')
    print(response)

    try:
        result = response.json()
        prediction_score = result['prediction'][0]

        # Classify as 'Credit accepted' if probability of class 0 is greater than 0.5
        if prediction_score > 0:
            prediction_result = 'Credit accepted'
        else:
            prediction_result = 'Credit denied'

        return prediction_result, prediction_score

    except Exception as e:
        st.error(f"Error getting prediction: {e}")
        return None, None

# Configuration de la page
st.set_page_config(
    page_title = 'Scoring crédit',
    page_icon = 'scoring_app/app_illustrations/pret_a_depenser_logo.png',
    layout = 'wide'
    )

# Définition des deux pages de l'application
st.sidebar.image('scoring_app/app_illustrations/pret_a_depenser_logo.png')
app_mode = st.sidebar.selectbox('Select Page', [
    'Home', # page d'accueil et description des variables
    'Prediction' # page pour faire les prédictions et expliquer le choix
    ])

if app_mode == 'Home':
    st.title('SCORING CREDIT')
    st.divider()
    st.subheader("Outil de scoring crédit pour calculer la probabilité qu’un client rembourse son crédit, et classifier la demande en crédit accordé ou refusé")
    st.divider()
    st.image('scoring_app/app_illustrations/paying-off-a-loan-early.jpg')

    # à défaut d'afficher infos sur jeu de données, décrire les features utilisées:
    st.subheader("Liste des descripteurs utilisés pour prédire l'attribution de prêts :")
    st.write("- :blue[CNT_CHILDREN]     :     Nombre d'enfants qu'à le/la client(e)")
    st.write("- :blue[CNT_FAM_MEMBERS]     :     Nombre de membres dans la famille")
    st.write("- :blue[PREVIOUS_LOANS_COUNT]     :     Nombre total des précédents crédits pris par chaque client")
    st.write("- :blue[NONLIVINGAREA_MODE]     :     Informations normalisées sur l'immeuble où vit le client (taille, étages, etc...)")
    st.write("- :blue[AMT_REQ_CREDIT_BUREAU_QRT]     :     Nombre de demandes de renseignements auprès du bureau de crédit concernant le client 3 mois avant la demande (à l'exclusion du mois précédant la demande)")
    st.write("- :blue[AMT_REQ_CREDIT_BUREAU_YEAR]     :     Nombre de demandes de renseignements auprès du bureau de crédit concernant le client sur un an (à l'exclusion des 3 derniers mois avant la demande)")
    st.write("- :blue[EXT_SOURCE_3]     :     Score normalisé provenant d'une source de données externe.")
    st.write("- :blue[OBS_30_CNT_SOCIAL_CIRCLE]     :     Combien d'observations des environs sociaux du client avec un défaut observable de 30 jours de retard (30 DPD).")
    st.write("- :blue[OBS_60_CNT_SOCIAL_CIRCLE]     :     Combien d'observations des environs sociaux du client avec un défaut observable de 60 jours de retard (30 DPD).")
    st.write("- :blue[DEF_30_CNT_SOCIAL_CIRCLE]     :     Combien d'observations des environs sociaux du client ont fait défaut avec un retard de paiement de 30 jours (30 DPD)")

elif app_mode == 'Prediction':
    st.image('scoring_app/app_illustrations/multi-currency-iban.jpg', width = 800)
    phrase = '''
    Bonjour,
    merci de remplir les informations suivantes à propos du client afin de déterminer si nous devons acceder à sa demande de prêt
    '''
    st.subheader(phrase)

    st.sidebar.header('Informations à propos du client:')

    childrennumber = st.sidebar.radio("Nombre d'enfants", options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) # CNT_CHILDREN = 'Number of children the client has'
    obs_30_cnt_social_circle = st.sidebar.slider("Nb d'observations de l'entourage du client (30 derniers jours)", 0, 350, 0) # OBS_30_CNT_SOCIAL_CIRCLE = "How many observation of client's social surroundings with observable 30 DPD (days past due) default"
    nonlivingarea_mode = st.sidebar.slider('Non living area mode (normalisation)', 0.0, 1.0, 0.01) # NONLIVINGAREA_MODE = 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor'
    ext_source_3 = st.sidebar.slider('Normalized score for external datasource', 0.0, 1.0, 0.01) # EXT_SOURCE_3 = 'Normalized score from external data source'
    def_30_cnt_social_circle = st.sidebar.slider("Nb d'observations de l'entourage du client (30 jours par défaut)", 0, 35, 0) # DEF_30_CNT_SOCIAL_CIRCLE = "How many observation of client's social surroundings defaulted on 30 DPD (days past due) "
    amt_req_credit_bureau_qrt = st.sidebar.slider('Nb enquêtes 3 derniers mois', 0, 300, 0) # AMT_REQ_CREDIT_BUREAU_QRT = 'Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application)'
    previous_loans_count = st.sidebar.slider('Nb de prêts antérieurs', 0, 150, 0) # PREVIOUS_LOANS_COUNT - nombre de prêts précedants
    amt_req_credit_bureau_year = st.sidebar.slider('Nb enquêtes année passée', 0, 30, 0) # AMT_REQ_CREDIT_BUREAU_YEAR = 'Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application)'
    obs_60_cnt_social_circle = st.sidebar.slider("Nb d'observations de l'entourage du client (60 jours par défaut)", 0, 350, 0) # OBS_60_CNT_SOCIAL_CIRCLE = "How many observation of client's social surroundings with observable 60 DPD (days past due) default"
    cnt_fam_members = st.sidebar.selectbox('Nb de membres dans la famille', (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)) # CNT_FAM_MEMBERS = 'How many family members does client have'

    # Données du client:
    data1 = {
         'CNT_CHILDREN' : childrennumber,
         'OBS_30_CNT_SOCIAL_CIRCLE' : obs_30_cnt_social_circle,
         'NONLIVINGAREA_MODE' : nonlivingarea_mode,
         'EXT_SOURCE_3' : ext_source_3,
         'DEF_30_CNT_SOCIAL_CIRCLE' : def_30_cnt_social_circle,
         'AMT_REQ_CREDIT_BUREAU_QRT' : amt_req_credit_bureau_qrt,
         'PREVIOUS_LOANS_COUNT' : previous_loans_count,
         'AMT_REQ_CREDIT_BUREAU_YEAR' : amt_req_credit_bureau_year,
         'OBS_60_CNT_SOCIAL_CIRCLE' : obs_60_cnt_social_circle,
         'CNT_FAM_MEMBERS' : cnt_fam_members 
         }

    # Lister les features dans le même ordre que graph shap :
    feature_list = [
        ext_source_3,
        obs_30_cnt_social_circle,
        amt_req_credit_bureau_qrt,
        obs_60_cnt_social_circle,
        nonlivingarea_mode,
        previous_loans_count,
        cnt_fam_members,
        def_30_cnt_social_circle,
        amt_req_credit_bureau_year,
        childrennumber
        ]

    single_sample = np.array(feature_list).reshape(1, -1)

    if st.button('Predict'):

        # faire la prédiction en appelant l'api
        print(feature_list)
        prediction = get_prediction(feature_list)

        if prediction[0] == 0:
            # Prêt rejeté
            file = open('scoring_app/app_illustrations/Loan-Rejection.jpg', 'rb')
            contents = file.read()
            data_url_no = base64.b64encode(contents).decode('utf-8')
            file.close()
            st.error('Selon notre prédiction, le prêt ne sera pas accordé')
            st.markdown(f'<img src="data:image/gif;base64, {data_url_no}" alt="cat gif">', unsafe_allow_html = True)

        elif prediction[0] == 1:
            # Prêt accepté
            file_ = open('scoring_app/app_illustrations/bank-loan-successfully-illustration-concept-white-background_701961-3161.avif', "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode('utf-8')
            file_.close()
            st.success('Selon notre prédiction, le prêt sera accordé')
            st.markdown(f'<img src="data:image/gif;base64, {data_url}" alt="cat gif">', unsafe_allow_html = True)

        st.divider()

        # Afficher l'explication de la prédiction (waterfall plot)
        explainer = pickle.load(open('utils/explainer', 'rb'))
        shap_values = explainer(single_sample)
        st.header('Explication de la prédiction:')
        fig, ax = plt.subplots(figsize = (10, 5))
        shap.waterfall_plot(shap_values[0])
        st.pyplot(fig)