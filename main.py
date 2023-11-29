from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from io import BytesIO
import pandas as pd
from utils import get_prediction

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.get("/")
def root():
    return 'My first raw model'


@app.post('/predict_items_via_csv')
def upload(file: UploadFile) -> FileResponse:
    # считываем байтовое содержимое
    content = file.file.read()
    # создаем буфер типа BytesIO
    buffer = BytesIO(content)
    # считываем в датафрейм
    csv_data = pd.read_csv(buffer)
    # закрывается именно сам файл
    buffer.close()
    file.close()
    # предсказание
    pred = get_prediction(csv_data)
    pred_df = pd.DataFrame(pred, columns=['pred_selling_price'])
    # cоединяем с изначальным датафреймом
    res = pd.concat([csv_data, pred_df], axis=1)
    # csv ответ
    res.to_csv('res.csv')
    response = FileResponse(path='res.csv', media_type='text/csv', filename='res_download.csv')

    return response


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    # получаем датафрейм
    input_df = pd.DataFrame.from_dict(item.model_dump(mode='json'), orient='index').T
    # предсказание
    pred = get_prediction(input_df)

    return pred[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    # получаем датафрейм
    input_df = pd.DataFrame()
    for item in items:
        input_df = pd.concat([input_df, pd.DataFrame.from_dict(item.model_dump(mode='json'), orient='index').T])
    # предсказание
    pred = get_prediction(input_df)

    return pred.tolist()
