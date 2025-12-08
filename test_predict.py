import requests
import json

url = 'http://127.0.0.1:5000/predict'
data = {
    "age": 60,
    "sex": 1,
    "cp": 0,
    "fbs": 0,
    "restecg": 1,
    "exang": 0,
    "slope": 2,
    "ca": 0,
    "thal": 2,
    "trestbps": 130,
    "chol": 250,
    "thalach": 150,
    "oldpeak": 1.5
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
