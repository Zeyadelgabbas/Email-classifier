from fastapi import FastAPI  , Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.schemas import Prediction_features
from src.pipeline.predict_pipeline import PredictPipeline , CustomData
import uvicorn


app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("temp.html", {"request": request})


@app.post("/predict")
async def predict_json(features: Prediction_features):
    features_dic = features.model_dump()
    custom_data = CustomData()
    df = custom_data.data_preparation(features_dic)
    predict = PredictPipeline()
    prediction = predict.predict(df)
    return {"prediction": prediction}