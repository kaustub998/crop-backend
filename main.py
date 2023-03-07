from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
from classes import id2label
import mysql.connector
from typing import List


app = FastAPI()

origins = [
    "https://merokhet.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

filename = "svm_model_pickle.pkl"
with open(filename, 'rb') as file:
    model = pickle.load(file)


mydb = mysql.connector.connect(
  host="sql12.freemysqlhosting.net",
  user="sql12603561",
  password="BIluCjww8i",
  database="sql12603561"
)

mycursor = mydb.cursor()

X_mean = np.array([50.547159,
 53.283523,
 48.150000,
 25.602018,
 71.437671,
 6.473384])

X_std = np.array([37.084931,
 33.008001,
 50.646067,
 5.090553,
 22.287511,
 0.770325])

class PhoneNumber(BaseModel):
    phone_no: str

class Node(BaseModel):
    node_id: str
    temp: str
    humidity: str
    pH: str

class Form(BaseModel):
    node: int
    checked: bool
    N: float
    P: float
    K: float

class FormList(BaseModel):
    data: List[Form]

data = {}

@app.get('/')
def req():
    return "Wassupp!!"

@app.post('/first-page')
def fetch_phone(item: PhoneNumber):
    phone_no = item.phone_no
    print(phone_no)

    mycursor = mydb.cursor()
    mycursor.execute('''select node_id as nodeID,pname as person_name,phone_no
    from nodes,person 
    where (nodes.person_id = person.id);''')
    ans = mycursor.fetchall()

    nodes = []
    for x in ans:
        if x[2] == phone_no:
            nodes.append(x[0])

    return {"nodes": nodes}

@app.post('/from-node')
def fetch_data(item: Node):
    dict = {}
    # data['node_id'] = int(item.node_id)
    dict['temp'] = float(item.temp)
    dict['humidity'] = float(item.humidity)
    dict['pH'] = float(item.pH)
    data[int(item.node_id)] = dict
    # print(item.node_id)
    # print(item.temp)
    # print(item.humidity)
    # print(item.pH)
    # print(data)

# data = {
#     1: [56, 67, 43],
#     2: [56, 67, 43],
#     3: [34, 87, 23],
#     4: [56, 67, 23]
# }

@app.post('/form-submit')
def submit(items: FormList):
    print(items)
    predictions = []
    for item in items.data:
        ans = {}
        prediction_crop = []
        dict = {}
        node_id = int(item.node)
        checked = item.checked

        if checked:
            dict['N'] = item.N
            dict['P'] = item.P
            dict['K'] = item.K
            # print(item.N)
            # print(item.P)
            # print(item.K)


            # mycursor = mydb.cursor()
            # mycursor.execute('''select node_id as nodeID,pname as person_name,phone_no
            # from nodes,person 
            # where (nodes.person_id = person.id);''')
            # ans = mycursor.fetchall()

            required_data = data[node_id]

    # data[int(item.phone_no)] = dict

            X = np.array([[dict['N'], dict['P'], dict['K'], required_data['temp'], required_data['humidity'], required_data['ph']]])
            # X = np.array([[dict['N'], dict['P'], dict['K'], 23, 34, 34]])

            X = (X - X_mean) / X_std

            # data = np.array([[-1.367733, -1.073527, -0.674368, 0.797281, 0.922049, 0.954140, 0.199674]])

            # data = np.array([[-0.417275, 0.615201, 0.515993, 0.214042, 0.375975, 0.558398, 1.101592]])
            prediction = model.predict(X)
            crop = id2label[prediction[0]].capitalize()
            prediction_crop.append(crop)

            ans['node'] = node_id
            ans['N'] = dict['N']
            ans['P'] = dict['P']
            ans['K'] = dict['K']
            ans['pH'] = required_data['pH']
            ans['temp'] = required_data['temp']
            ans['hum'] = required_data['humidity']
            # ans['pH'] = 5
            # ans['temp'] = 23
            # ans['hum'] = 34
            ans['predictions'] = prediction_crop

            predictions.append(ans)

    # print(predictions) 
    return {"result": predictions}
