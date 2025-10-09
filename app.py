# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from io import BytesIO

app = FastAPI(title="Alphabet Sign Language Detection API")

# Load model and processor once (to speed up inference)
model_name = "prithivMLmods/Alphabet-Sign-Language-Detection"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label map
labels = {
    "0": "A", "1": "B", "2": "C", "3": "D", "4": "E", "5": "F", "6": "G", "7": "H", "8": "I", "9": "J",
    "10": "K", "11": "L", "12": "M", "13": "N", "14": "O", "15": "P", "16": "Q", "17": "R", "18": "S", "19": "T",
    "20": "U", "21": "V", "22": "W", "23": "X", "24": "Y", "25": "Z"
}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sign Language Detection API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Process and predict
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()

        # Build predictions dict
        predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
        top_prediction = max(predictions, key=predictions.get)

        return JSONResponse(content={
            "predicted_letter": top_prediction,
            "confidence": predictions[top_prediction],
            "all_predictions": predictions
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
