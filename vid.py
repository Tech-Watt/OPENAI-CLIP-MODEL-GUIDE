import cv2
import os

# Create the 'images' directory
directory = 'images'
os.makedirs(directory, exist_ok=True)
if os.path.exists(directory):
    print(f"Directory '{directory}' created successfully.")
else:
    print(f"Failed to create directory '{directory}'.")

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = total_frames / frame_rate
    target_frame_count = int(total_seconds)
    target_frame_index = 0
    frame_index = 0

    while frame_index < target_frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        # Generate a unique filename for each frame
        filename = f"img{frame_index}.jpg"
        filepath = os.path.join(directory, filename)
        cv2.imwrite(filepath, frame)

        target_frame_index += int(frame_rate)
        frame_index += 1
        print(target_frame_index, target_frame_count, sep=' ')

    cap.release()

# Call the function with the video path
extract_frames('video2.mp4')