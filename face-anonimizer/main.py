import os
import argparse
import numpy as np

import cv2
import mediapipe as mp


def process_img(img, face_detection):
    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box

            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)

            # Face center
            cx = x1 + w // 2
            cy = y1 + h // 2

            # Oval radii — faces are taller than wide (~1:1.3 ratio)
            rx = int(w * 0.50)  # width radius
            ry = int(h * 0.60)  # height radius

            # Blur image
            blurred = cv2.GaussianBlur(img, (75, 75), 50)

            # Create empty mask
            mask = np.zeros((H, W), dtype=np.uint8)

            # Draw ellipse (oval)
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

            # 3-channel mask
            mask_3c = cv2.merge([mask, mask, mask])

            # Blend: where mask = blur, else original
            img = np.where(mask_3c == 255, blurred, img)

    return img


args = argparse.ArgumentParser()

args.add_argument("--mode", default="webcam")
args.add_argument("--filePath", default=None)

args = args.parse_args()


output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
) as face_detection:
    if args.mode in ["image"]:
        # read image
        img = cv2.imread(args.filePath)

        img = process_img(img, face_detection)

        # save image
        cv2.imwrite(os.path.join(output_dir, "output.png"), img)

    elif args.mode in ["video"]:
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_video = cv2.VideoWriter(
            os.path.join(output_dir, "output.mp4"),
            cv2.VideoWriter_fourcc(*"MP4V"),
            25,
            (frame.shape[1], frame.shape[0]),
        )

        while ret:
            frame = process_img(frame, face_detection)

            output_video.write(frame)

            ret, frame = cap.read()

        cap.release()
        output_video.release()

    elif args.mode in ["webcam"]:
        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()
        while ret:
            frame = process_img(frame, face_detection)

            cv2.imshow("frame", frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
            ret, frame = cap.read()

        cap.release()
