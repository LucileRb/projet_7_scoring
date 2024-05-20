#### IMPORTS #####
import base64
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt


@st.cache(suppress_st_warning = True)

def get_fvalue(val):
    feature_dict = {'No' : 1 ,'Yes' : 2}
    for key, value in feature_dict.items():
        if val == key:
            return value

def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Prediction']) #two pages


if app_mode == 'Home':
    st.title('LOAN PREDICTION :')
    st.image('scoring_app/app_illustrations/paying-off-a-loan-early.jpg')
    st.markdown('Dataset :')

    path = ''
    filepath = os.path.join(path, '../data/df_merged')
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    st.write(data.head())
    st.markdown('Applicant Income VS Loan Amount ')
    st.bar_chart(data[['AMT_INCOME_TOTAL', 'AMT_CREDIT']].head(20))

    st.markdown("Visualiser les données d'un prêt:")
    loan_id = txt = st.text_input("Entrez l'ID du prêt...")
    if st.button('Voir les données'):
        # faire la prédiction en utilisant le modèle entrainé
        st.write(data.loc[data['SK_ID_CURR'] == loan_id])

elif app_mode == 'Prediction':
    st.image('scoring_app/app_illustrations/multi-currency-iban.jpg')
    st.subheader('Bonjour, merci de remplir les informations suivantes à propos du client afin de déterminer si nous devons acceder à sa demande de prêt:')
    st.sidebar.header('Informations à propos du client:')


    childrennumber = st.sidebar.radio("Nombre d'enfants", options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # CNT_CHILDREN = 'Number of children the client has'
    obs_30_cnt_social_circle = st.sidebar.slider("Nb d'observations de l'entourage du client (30 derniers jours)", 0, 5, 0) # OBS_30_CNT_SOCIAL_CIRCLE = "How many observation of client's social surroundings with observable 30 DPD (days past due) default"
    nonlivingarea_mode = st.sidebar.slider('Non living area mode (normalisation)', 0.0, 10.0, 0.1) # NONLIVINGAREA_MODE = 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor'
    ext_source_3 = st.sidebar.slider('Normalized score for external datasource', 0.0, 1.0, 0.1) # EXT_SOURCE_3 = 'Normalized score from external data source'
    def_30_cnt_social_circle = st.sidebar.slider("Nb d'observations de l'entourage du client (30 jours par défaut)", 0, 5, 0) # DEF_30_CNT_SOCIAL_CIRCLE = "How many observation of client's social surroundings defaulted on 30 DPD (days past due) "
    amt_req_credit_bureau_qrt = st.sidebar.slider('Nb enquêtes 3 derniers mois', 0, 5, 0) # AMT_REQ_CREDIT_BUREAU_QRT = 'Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application)'
    previous_loans_count = st.sidebar.radio('Nb de prêts antérieurs', options = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # PREVIOUS_LOANS_COUNT - nombre de prêts précedants
    amt_req_credit_bureau_year = st.sidebar.slider('Nb enquêtes année passée', 0, 5, 0) # AMT_REQ_CREDIT_BUREAU_YEAR = 'Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application)'
    obs_60_cnt_social_circle = st.sidebar.slider("Nb d'observations de l'entourage du client (60 jours par défaut)", 0, 5, 0) # OBS_60_CNT_SOCIAL_CIRCLE = "How many observation of client's social surroundings with observable 60 DPD (days past due) default"
    cnt_fam_members = st.sidebar.selectbox('Nb de membres dans la famille', (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)) # CNT_FAM_MEMBERS = 'How many family members does client have'

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

    feature_list = [
        childrennumber,
        obs_30_cnt_social_circle,
        nonlivingarea_mode,
        ext_source_3,
        def_30_cnt_social_circle,
        amt_req_credit_bureau_qrt,
        previous_loans_count,
        amt_req_credit_bureau_year,
        obs_60_cnt_social_circle,
        cnt_fam_members
        ]

    single_sample = np.array(feature_list).reshape(1, -1)

    if st.button('Predict'):
        # faire la prédiction en utilisant le modèle entrainé
        loaded_model = pickle.load(open('../best_model', 'rb'))
        prediction = loaded_model['classification'].predict(single_sample)

        if prediction[0] == 0:
            # Prêt rejeté
            file = open('app_illustrations/Loan-Rejection.jpg', 'rb')
            contents = file.read()
            data_url_no = base64.b64encode(contents).decode('utf-8')
            file.close()
            st.error('Selon notre prédiction, le prêt ne sera pas accordé')
            st.markdown(f'<img src="data:image/gif;base64, {data_url_no}" alt="cat gif">', unsafe_allow_html = True)

        elif prediction[0] == 1:
            # Prêt accepté
            file_ = open('app_illustrations/bank-loan-successfully-illustration-concept-white-background_701961-3161.avif', "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode('utf-8')
            file_.close()
            st.success('Selon notre prédiction, le prêt sera accordé')
            st.markdown(f'<img src="data:image/gif;base64, {data_url}" alt="cat gif">', unsafe_allow_html = True)


        # Afficher l'explication de la prédiction (waterfall plot)
        explainer = pickle.load(open('../explainer', 'rb'))
        shap_values = explainer(single_sample)
        st.header('Explication de la prédiction:')
        fig, ax = plt.subplots(figsize = (10, 5))
        shap.waterfall_plot(shap_values[0])
        st.pyplot(fig)