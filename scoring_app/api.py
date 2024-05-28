# Imports
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify

# Créer une application Flask
app = Flask(__name__)

# Importer modèle entrainé
with open('utils/best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Importer scaler entrainé -> entraîné avec 175 features alors que là seulement 10
#with open('utils/scaler.pkl', 'rb') as scaler_file:
#    scaler = pickle.load(scaler_file)


@app.route('/')
def home_page():
    return 'Welcome to the credit scoring API'

# Dans onglet "Prediction" du site, méthode post pour envoyer une information à l'api (data) et récupérer prédiction
@app.route('/Prediction', methods = ['POST'])
def predict():
    try:
        # récupérer les données
        print('predict api')
        data = request.get_json(force = True)
        print(f'data: {data}')
        test_data = np.array(data).reshape(1, -1)
        print(f'test_data: {test_data}')

        # scaling des données
        scaler = MinMaxScaler(feature_range = (0, 1))
        scaler.fit(test_data)
        scaled_data = scaler.transform(test_data)
        print(f'scaled_data: {scaled_data}')

        # predict
        prediction = model.predict(scaled_data)
        #prediction = model.predict_proba(scaled_data)[:, 0]
        print(f'prediction: {prediction}')

        return jsonify({'prediction': prediction.tolist()}) # return le résultat dans un dictionnaire - tolist car ne prend pas les np arrays

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host = '0.0.0.0', port = port)