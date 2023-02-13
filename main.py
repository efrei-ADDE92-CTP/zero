from fastapi import FastAPI, Request
import joblib, json
from prometheus_client import Counter,generate_latest ,Summary
import time
from fastapi.responses import JSONResponse

app = FastAPI()

from sklearn.datasets import load_iris 
from prometheus_client.parser import text_string_to_metric_families
from sklearn.model_selection import train_test_split

#Get testing data
iris = load_iris() 
X = iris.data 
y = iris.target
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.3, random_state = 2020)

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')

c = Counter("Classifier_request_count", "Number of requests processed")



@REQUEST_TIME.time()
@app.post("/predict-iris")
async def get_iris(info : Request):
    data = await info.json()
    print("Data:", data)

   
    sepal_length = data["sepal_length"]
    sepal_width = data["sepal_width"]
    petal_length = data["petal_length"]
    petal_width = data["petal_width"]

    iris_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    print("Iris data:", iris_data)

    iris_model = joblib.load('iris_classification_model.pkl')

    pred_result = iris_model.predict(iris_data)

    print(pred_result)
    time.sleep(1)
    c.inc()
    return json.dumps(pred_result.tolist())


@app.get('/metrics')
async def metrics():
    return generate_latest()


#Launch API : uvicorn main:app --reload