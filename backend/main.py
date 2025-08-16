from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import uvicorn


# Define request schema
class NewsInput(BaseModel):
    text: str

# Initialize app
app = FastAPI(title="Fake News Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("fake_news_pred_model1.pkl")
vectorizer = joblib.load("fake_news_vectorizer_model1.pkl")  # if you saved a TF-IDF vectorizer too

@app.get("/")
def home():
    return {"message": "Fake News Detector API is running!"}

@app.post("/predict")
def predict(input: NewsInput):
    # Preprocess
    vectorized_text = vectorizer.transform([input.text])
    prediction = model.predict(vectorized_text)[0]

    return {
        "text": input.text,
        "prediction": "FAKE" if prediction == 1 else "REAL"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)