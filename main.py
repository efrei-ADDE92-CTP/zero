from fastapi import FastAPI, Request
import joblib, json
from prometheus_client import Counter,generate_latest ,Summary
import time
from fastapi.responses import JSONResponse
from starlette_exporter import PrometheusMiddleware, handle_metrics


app = FastAPI()
app = FastAPI()
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)




REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')

c = Counter("Classifier_request_count", "Number of requests processed")

@REQUEST_TIME.time()
@app.post("/predict-iris")
async def get_iris(info : Request):
    data = await info.json()
   
    sepal_length = data["sepal_length"]
    sepal_width = data["sepal_width"]
    petal_length = data["petal_length"]
    petal_width = data["petal_width"]

    iris_data = [[sepal_length, sepal_width, petal_length, petal_width]]

    #Load model
    iris_model = joblib.load('iris_classification_model.pkl')

    pred_result = iris_model.predict(iris_data)

    print(pred_result)
    c.inc()
    return json.dumps(pred_result.tolist())


@app.get('/metrics')
async def metrics():    
    return generate_latest()

#Launch API : uvicorn main:app --reload