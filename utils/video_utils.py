import cv2

def extract_frames(video_path, skip_frames=4):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames
