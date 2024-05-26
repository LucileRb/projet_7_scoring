import unittest
import json
from api import app
import pandas as pd
import sklearn


class FlaskAppTest(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.app = app.test_client()

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode('utf-8'), 'Welcome to the credit scoring API')

    def test_predict_endpoint(self):

        test_data = pd.read_parquet('./utils/test_data.parquet')
        test_data = test_data.sample(n = 1).drop('SK_ID_CURR', axis=1)
        test_data = {'test_data': test_data.values.tolist()}
        headers = {'Content-Type': 'application/json'}

        response = self.app.post('/predict', data=json.dumps(test_data), headers=headers)
        data = json.loads(response.data.decode('utf-8'))

        self.assertEqual(response.status_code, 200)
        self.assertTrue('prediction' in data)

if __name__ == '__main__':
    unittest.main()