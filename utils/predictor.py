import numpy as np
import cv2
import keras as keras

model_path = r"C:\Users\Admin\code\signspeak-api\assets\american-sign-language-tensorflow2-american-sign-language-v1"
model = keras.layers.TFSMLayer(
    model_path,
    call_endpoint="serving_default"
)

# Preprocess input frames
def preprocess_frame(frame):
    resized = cv2.resize(frame, (224, 224))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)

# Decode prediction index to letter (A-Z)
def decode_prediction(prediction):
    class_index = np.argmax(prediction)
    class_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return class_labels[class_index]

# Predict letters for list of frames
def predict_alphabets(frames):
    predictions = []
    consistency_counter = 0
    consistency_threshold = 3
    buffer = []

    for frame in frames:
        input_tensor = preprocess_frame(frame)
        output_dict = model(input_tensor)
        output_tensor = list(output_dict.values())[0]
        output = output_tensor.numpy()

        predicted_label = decode_prediction(output)
        buffer.append(predicted_label)

        if len(buffer) >= consistency_threshold:
            # Take the most common label in buffer
            most_common_label = max(set(buffer), key=buffer.count)
            predictions.append(most_common_label)
            buffer = []  # Reset buffer after recording a prediction

    return predictions