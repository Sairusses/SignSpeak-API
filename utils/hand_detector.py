import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

def process_frames_with_mediapipe(frames):
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    hand_frames = []

    for frame in frames:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_frames.append(image_rgb)

    hands.close()
    return hand_frames
