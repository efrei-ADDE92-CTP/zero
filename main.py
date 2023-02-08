from fastapi import FastAPI, Request
import joblib, json
from prometheus_client import Counter
from prometheus_client.core import CollectorRegistry
import prometheus_client


app = FastAPI()

from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split

#Get testing data
iris = load_iris() 
X = iris.data 
y = iris.target
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.3, random_state = 2020)

@app.post("/predict-iris")
async def get_iris(info : Request):
    data = await info.json()
    print("Data:", data)

    c.inc()
    sepal_length = data["sepal_length"]
    sepal_width = data["sepal_width"]
    petal_length = data["petal_length"]
    petal_width = data["petal_width"]

    iris_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    print("Iris data:", iris_data)

    iris_model = joblib.load('iris_classification_model.pkl')

    pred_result = iris_model.predict(iris_data)

    print(pred_result)


    return json.dumps(pred_result.tolist())

c = Counter("Classifier_g4_request_count", "Number of requests processed")

@app.get("/metrics")
def metrics():
    res = []
    res.append(prometheus_client.generate_latest(c))
    return res

#Launch API : uvicorn main:app --reload