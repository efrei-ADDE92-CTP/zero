from fastapi import FastAPI, Request
import joblib, json
from prometheus_client import Counter,generate_latest ,Summary
import time
from fastapi.responses import JSONResponse

app = FastAPI()

from sklearn.datasets import load_iris 
from prometheus_client.parser import text_string_to_metric_families
from sklearn.model_selection import train_test_split

#Load model
iris_model = joblib.load('iris_classification_model.pkl')


REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')

c = Counter("Classifier_request_count", "Number of requests processed")

def format_string(s):
    lines = s.split("\n")
    return "\n".join(lines)


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


    pred_result = iris_model.predict(iris_data)

    print(pred_result)
    c.inc()
    return json.dumps(pred_result.tolist())


@app.get('/metrics')
async def metrics():    
    metrics = str(generate_latest().decode())
    print(metrics)
    return metrics


#Launch API : uvicorn main:app --reload