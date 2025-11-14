import os
import cv2
import numpy as np
from utils import get_face_landmarks

data_dir = "./facial_emotion_dataset/"
output_file = "data.txt"

with open(output_file, "w") as f_out:
    for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
        for image_path_ in os.listdir(os.path.join(data_dir, emotion)):
            image_path = os.path.join(data_dir, emotion, image_path_)
            image = cv2.imread(image_path)

            face_landmarks = get_face_landmarks(image)

            if len(face_landmarks) == 1404:
                face_landmarks.append(int(emotion_indx))
                line = ",".join(map(str, face_landmarks)) + "\n"
                f_out.write(line)
