# Imports
import pytest
import json
from flask import Flask

# Importer l'application Flask
from api import app

# Configurer pytest pour utiliser l'application Flask
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test de la route home_page"""
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'Welcome to the credit scoring API' in rv.data

def test_prediction_valid_input(client):
    """Test de la route prediction avec une entrée valide"""
    data = [0.1] * 10  # Assurez-vous que cela correspond au nombre de caractéristiques attendues
    rv = client.post('/Prediction', data = json.dumps(data), content_type = 'application/json')
    assert rv.status_code == 200
    response_data = json.loads(rv.data)
    assert 'prediction' in response_data

def test_prediction_invalid_input(client):
    """Test de la route prediction avec une entrée invalide"""
    data = "invalid input"
    rv = client.post('/Prediction', data = json.dumps(data), content_type = 'application/json')
    assert rv.status_code == 200
    response_data = json.loads(rv.data)
    assert 'error' in response_data

# Exécuter les tests avec pytest
if __name__ == '__main__':
    pytest.main()
