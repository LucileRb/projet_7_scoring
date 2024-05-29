#### IMPORTS #####
import base64
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import requests
import pyarrow.parquet as pq
import json

################################################ FONCTIONS ################################################

def get_prediction(data):
    api_url = 'http://127.0.0.1:5000/Prediction' # url de l'api sur heroku
    response = requests.post(api_url, json = data)

    try:
        result = response.json()
        prediction_score = result['prediction'][0]

        # Classify as 'Credit accepted' if probability of class 0 is greater than 0.5
        if prediction_score > 0.55:
            prediction_result = 'Credit accepted'
        else:
            prediction_result = 'Credit denied'

        return prediction_result, prediction_score

    except Exception as e:
        st.error(f"Error getting prediction: {str(e)}")


def credit_score_gauge(score):
    # Color gradient from red to yellow to green
    colors = ['#FF0000', '#FFFF00', '#00FF00']  # Red, Yellow, Green
    thresholds = [0, 0.5, 1]

    # Interpolate color based on score
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", list(zip(thresholds, colors)))
    norm = mcolors.Normalize(vmin = 0, vmax = 1)
    #color = cmap(norm(score))

    # Plot gauge
    fig, ax = plt.subplots(figsize = (6, 0.5))  # Reduced height to accommodate lower text
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Draw color gradient as colorbar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect = 'auto', cmap = cmap, extent = [0, 1, 0, 0.5])

    # Draw tick marks and labels
    for i, threshold in enumerate(thresholds):
        ax.plot([threshold, threshold], [0.45, 0.5], color = 'black')
        ax.text(threshold, 0.55, str(threshold), fontsize = 12, ha = 'center', va = 'bottom', color = 'black')

    # Draw dotted line at 0.5 threshold with legend
    ax.plot([0.5, 0.5], [0, 0.5], linestyle = '--', color = 'black', label = 'Threshold')
    # Draw prediction indicator with legend
    ax.plot([score, score], [0, 0.5], color = 'black', linewidth = 2, label = 'Client score')
    # Draw score below with the same color as the prediction indicator
    ax.text(score, -0.7, f'{score:.2f}', fontsize = 14, ha = 'center', va = 'bottom', color = 'black')
    # Add legend
    ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.8), fancybox = True, shadow = True, ncol = 2)

    st.pyplot(fig, clear_figure = True)

# Function to visualize client features
def visualize_client_features(selected_client_data, selected_feature):
    # Display position of client among others
    client_value = selected_client_data[selected_feature].values[0]
    st.text(f'Client {selected_feature}: {client_value:.2f}')

    # Plot client position in distribution
    fig, ax = plt.subplots()

    # Filter DataFrame based on prediction result
    filtered_df = df_train[df_train['TARGET'] == int(prediction_result == 'Credit denied')]

    # Check if the selected feature is categorical or continuous
    if df_train[selected_feature].dtype == 'int64':  # Categorical feature
        sns.countplot(data = filtered_df, x = selected_feature, ax = ax)
        ax.axvline(x=np.where(filtered_df[selected_feature].unique() == client_value)[0][0], color = 'red', linestyle = '--', label = f'Client {selected_feature}')
        ax.set_title(f'Client Position in {selected_feature} Distribution')
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Count')
    else:  # Continuous feature
        sns.histplot(filtered_df[selected_feature], kde =True, ax = ax)
        ax.axvline(x = client_value, color = 'red', linestyle = '--', label = f'Client {selected_feature}')
        ax.set_title(f'Client Position in {selected_feature} Distribution')
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Density')

    ax.legend()
    st.pyplot(fig, clear_figure = True)

# Function for bivariate analysis
def bivariate_analysis(feature1, feature2):
    fig, ax = plt.subplots()

    filtered_df = df_train[df_train['TARGET'] == int(prediction_result == 'Credit denied')]

    # Check the types of the selected features
    if filtered_df[feature1].dtype == 'int64' and filtered_df[feature2].dtype == 'int64':  # Categorical vs Categorical
        sns.countplot(data = filtered_df, x = feature1, hue = feature2, ax = ax)
        ax.set_title(f'Bivariate Analysis between {feature1} and {feature2}')
        ax.set_xlabel(feature1)
        ax.set_ylabel('Count')
        ax.legend(title = feature2)
    elif filtered_df[feature1].dtype != 'int64' and filtered_df[feature2].dtype != 'int64':  # Continuous vs Continuous
        sns.scatterplot(data = filtered_df, x = feature1, y = feature2, ax = ax)
        ax.set_title(f'Bivariate Analysis between {feature1} and {feature2}')
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
    else:  # Continuous vs Categorical or Categorical vs Continuous
        if filtered_df[feature1].dtype == 'int64':  # Categorical vs Continuous
            categorical_feature, continuous_feature = feature1, feature2
        else:  # Continuous vs Categorical
            categorical_feature, continuous_feature = feature2, feature1
        sns.boxplot(data = filtered_df, x = categorical_feature, y = continuous_feature, ax = ax)
        ax.set_title(f'Bivariate Analysis between {categorical_feature} and {continuous_feature}')
        ax.set_xlabel(categorical_feature)
        ax.set_ylabel(continuous_feature)

    st.pyplot(fig, clear_figure = True)

# Function to visualize SHAP values for the selected client
def visualize_shap_values(selected_client_data):
    st.write("Features that contribute the most to the score globally")
    # Plot global contribution
    #fig, ax = plt.subplots()
    #plt.sca(ax)
    #shap.plots.bar(shap_values, show=False, max_display=10)
    #plt.title('Global SHAP Values Analysis')
    #st.pyplot(fig)

    # Chemin vers votre image localement
    chemin_image = "utils/global_score.png"
    # Afficher l'image
    st.image(chemin_image, caption = '', use_column_width = True)

    # Plot local contribution
    st.write("Features that contribute the most to the score for the selected client")
    fig, ax = plt.subplots()
    plt.sca(ax)
    shap_values_client = explainer(scaler.transform(selected_client_data.drop(columns = ['SK_ID_CURR'])), max_evals = 1000)
    # Plot local contribution
    force_plot_img = shap.plots.force(shap_values_client, matplotlib = True, show = False, contribution_threshold = 0.07, feature_names = selected_client_data.drop(columns = ['SK_ID_CURR']).columns)
    st.pyplot(force_plot_img, clear_figure = True)



################################################ DATA & OUTILS ################################################


# Load sample parquet data
parquet_file = 'utils/X_test_SS.parquet'
df = pq.read_table(parquet_file).to_pandas().reset_index(drop = True)

parquet_file_train = 'utils/X_train_SS.parquet'
df_train = pq.read_table(parquet_file_train).to_pandas().reset_index(drop = True)

with open('utils/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

explainer = pickle.load(open('utils/explainer', 'rb'))


features = [
    'CNT_CHILDREN',
    'OBS_30_CNT_SOCIAL_CIRCLE',
    'NONLIVINGAREA_MODE',
    'EXT_SOURCE_3',
    'DEF_30_CNT_SOCIAL_CIRCLE',
    'AMT_REQ_CREDIT_BUREAU_QRT',
    'PREVIOUS_LOANS_COUNT',
    'AMT_REQ_CREDIT_BUREAU_YEAR',
    'OBS_60_CNT_SOCIAL_CIRCLE',
    'CNT_FAM_MEMBERS'
    ]


################################################ DASHBOARD ################################################


# Configuration de la page
st.set_page_config(
    page_title = 'Scoring crédit',
    page_icon = 'scoring_app/app_illustrations/pret_a_depenser_logo.png',
    layout = 'wide'
    )

# Définition des deux pages de l'application
st.sidebar.image('scoring_app/app_illustrations/pret_a_depenser_logo.png')
app_mode = st.sidebar.selectbox('Select Page', [
    'Home', # page d'accueil et description des variables - description des données aussi ? à voir
    'Vue client', # page pour visualiser les informations descriptives relatives à un client (système de filtre) -> filtrer par ID client et visualiser les données du client
    'New prediction' # page pour faire les prédictions et expliquer le choix
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


elif app_mode == 'Vue client':
    # Dropdown for client IDs in the sidebar
    selected_client_id = st.sidebar.selectbox('Select Client ID:', df.index.tolist())

    # Display selected client's data in the main section
    st.sidebar.header('Selected Client Data:')
    selected_client_data = df[features].loc[df.index == selected_client_id]
    st.sidebar.write(selected_client_data)

    single_sample = np.array(selected_client_data).reshape(1, -1)
    json_data = json.dumps(single_sample.tolist())

    # Button to trigger prediction in the sidebar
    if st.sidebar.button('Predict'):
        # Make API request and get prediction
        prediction_result, prediction_score = get_prediction(json_data)

        # Display prediction result
        st.sidebar.subheader('Prediction Result:')
        if prediction_result is not None:
            # Determine emoji based on prediction result
            emoji = "❌" if prediction_result == "Credit denied" else "✅"

            # Display prediction result with emoji
            st.sidebar.write(f"{emoji} The credit is accepted if the score is greater than 0.5 or 50%, denied otherwise. In this case, the predicted score is {prediction_score:.2}")

            st.sidebar.write(f"{emoji} The credit status is: {prediction_result}")
            st.sidebar.write(f"{emoji} The prediction score is: {prediction_score:.2%}")
            st.sidebar.write(f"{emoji} The probability is: {prediction_score:.2}")


            # Visualisation du score de crédit (jauge colorée)
            st.subheader('Credit Score Visualization:')
            credit_score_gauge(prediction_score)
            st.text("A color gauge representing the credit score. The client's score is indicated by a marker on the gauge.")

            # Visualisation de la contribution des features
            st.subheader('Feature Contribution:')
            visualize_shap_values(selected_client_data)
            st.text("Bar chart and force plot showing the features that contribute the most to the credit score globally and for the selected client.")

            # Dropdown for feature selection
            selected_feature = st.selectbox('Select Feature:', df.drop(columns = ['SK_ID_CURR']).columns, key = 'feature_selection')
            st.text("A graphical representation of the client's position among others based on the selected feature for the same target as the client.")

            # Display client features visualization
            visualize_client_features(selected_client_data, selected_feature)

            # Graphique d’analyse bi-variée entre deux features sélectionnées
            st.subheader('Bi-variate Analysis:')
            # Select two features for bivariate analysis
            selected_feature1 = st.selectbox('Select Feature 1:', df.drop(columns = ['SK_ID_CURR']).columns, key = 'feature_selection1')
            selected_feature2 = st.selectbox('Select Feature 2:', df.drop(columns = ['SK_ID_CURR']).columns, key = 'feature_selection2')

            # Display bivariate analysis
            bivariate_analysis(selected_feature1, selected_feature2)
            st.text("A graphical analysis of the relationship between two selected features for the same target as the client.")


elif app_mode == 'New prediction':
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
    json_data = json.dumps(single_sample.tolist())

    if st.button('Predict'):

        # faire la prédiction en appelant l'api
        prediction_result, prediction_score = get_prediction(json_data)
        print(f'prediction: {prediction_score}')

        # Classify as 'Credit accepted' if probability of class 0 is greater than 0.5
        if prediction_result == 'Credit accepted':
            # Prêt accepté
            print('pret accepté')
            file_ = open('scoring_app/app_illustrations/bank-loan-successfully-illustration-concept-white-background_701961-3161.avif', "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode('utf-8')
            file_.close()
            st.success('Selon notre prédiction, le prêt sera accordé')
            st.markdown(f'<img src="data:image/gif;base64, {data_url}" alt="cat gif">', unsafe_allow_html = True)
        else:
            print('pret rejeté')
            # Prêt rejeté
            file = open('scoring_app/app_illustrations/Loan-Rejection.jpg', 'rb')
            contents = file.read()
            data_url_no = base64.b64encode(contents).decode('utf-8')
            file.close()
            st.error('Selon notre prédiction, le prêt ne sera pas accordé')
            st.markdown(f'<img src="data:image/gif;base64, {data_url_no}" alt="cat gif">', unsafe_allow_html = True)

        st.divider()

        # Afficher l'explication de la prédiction (waterfall plot)
        
        shap_values = explainer(single_sample)
        st.header('Explication de la prédiction:')
        fig, ax = plt.subplots(figsize = (10, 5))
        shap.waterfall_plot(shap_values[0])
        st.pyplot(fig)