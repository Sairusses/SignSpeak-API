from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from utils.video_utils import extract_frames
from utils.hand_detector import process_frames_with_mediapipe
from utils.predictor import predict_alphabets
from utils.nlp_processor import generate_tagalog_sentence

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "server started"}

@app.post("/predict/")
async def predict_sign_language(video: UploadFile = File(...)):

    # Save uploaded video
    os.makedirs("temp", exist_ok=True)
    temp_video_path = os.path.join("temp", video.filename)
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer) # buffer Expected type 'SupportsWrite[bytes]' (matched generic type 'SupportsWrite[AnyStr ≤: str | bytes]'), got 'BufferedWriter' instead

    # Step 1: Extract frames
    frames = extract_frames(temp_video_path)

    # Step 2: MediaPipe hand detection
    hand_frames = process_frames_with_mediapipe(frames)

    # Step 3: Predict ASL letters
    predicted_letters = predict_alphabets(hand_frames)

    # Step 4: NLP for Tagalog sentence
    tagalog_sentence = generate_tagalog_sentence(predicted_letters)

    # Clean up
    os.remove(temp_video_path)

    return JSONResponse({
        "raw_prediction": predicted_letters,
        "tagalog_sentence": tagalog_sentence
    })
