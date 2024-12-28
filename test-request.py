#if you deployed ML model in server thorugh Flask or FastAPI, how can you test it
# that your deployed model is processing everyhting successfully
#this tesp-requst.py help 
import requests
import json

#test data
test_data = {
    "age": 39,
    "workclass": 7,
    "education_num": 13,
    "marital_status": 1,
    "occupation": 4,
    "relationship": 1,
    "race": 4,
    "sex": 1,
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": 39

}

#send POST request for Flask
#for Fast api, you can send http request
response = requests.post(
    "http://localhost:5000/predict",
    json=test_data,
    headers={"Content-type":"application/json"}
)

#print results
print("Staus Code:", response.status_code)
print("Response:", response.json())