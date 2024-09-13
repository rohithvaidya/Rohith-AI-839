#mlflow models serve -m "runs:/de76a974868344329bafa0975bf29f24/model" -p 8001 --no-conda

import requests
import json

# URL for the MLflow model server
url = 'http://127.0.0.1:8001/invocations'

# Define the input data
data = {
    "instances": [list(X_test.values[5])]
}

# Send the POST request
response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))

# Print the prediction response
print(response.json())