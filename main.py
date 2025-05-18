from fastapi import FastAPI, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import joblib
import numpy as np

from fastapi.responses import JSONResponse
# from src.limiter import limiter
from src import limiter



@asynccontextmanager
async def load_model(app: FastAPI):
    print(f"Загружаем МЛ модель")
    global model
    model = joblib.load('research.joblib')
    yield
    print(f"Завершаем работу")

app=FastAPI(
    root_path="/api",
    lifespan=load_model,
    title="Документация к МЛ модели рассчета стоимости страховки",
    description="Здесь можно написать что-то более подробное о твеих эндпоинтах",
    version="v1.0",
    contact={
        "name": "Alexey A.",
        "email": "example@example.com"
    }
)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["localhost", "127.0.0.1", "0.0.0.0"], # << здесь указываем адреса с которых можно делать к нам запросы
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

class incomedata(BaseModel):
    age:int
    sex:int
    bmi:float
    children:int
    smoker:int
    region:int

@app.post('/predict')
@limiter.limit("1/10second", error_message="Слишко часто долбишься сюда, дружок :(")
async def predict(data:incomedata, request: Request):
    dictionary = {'age':data.age, 'sex':data.sex, 'bmi':data.bmi, 'children':data.children, 'smoker': data.smoker, 'region':data.region}
    # print(np.array([data.age, data.sex, data.bmi, data.children, data.smoker, data.region]).reshape(1,-1))
    # print(type(data.age))
    m = model.predict(np.array([data.age, data.sex, data.bmi, data.children, data.smoker, data.region]).reshape(1,-1))
    # print(m[0])
    return JSONResponse(content = {'insurance_amount':m[0]}, status_code = status.HTTP_200_OK)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True # FOR DEV ONLY
    )

    