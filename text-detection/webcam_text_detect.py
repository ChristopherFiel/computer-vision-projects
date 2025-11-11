import cv2
import pytesseract

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 170)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")

cntr = 0
while True:
    ret, frame = cap.read()
    cntr = cntr + 1
    if (cntr % 20) == 0:
        imgH, imgW, _ = frame.shape
        x1, y1, w1, h1 = 0, 0, imgH, imgW
        img2char = pytesseract.image_to_string(frame).strip()
        img2box = pytesseract.image_to_boxes(frame)

        for boxes in img2box.splitlines():
            boxes = boxes.split(" ")
            x, y, w, h = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
            cv2.rectangle(frame, (x, imgH - y), (w, imgH - h), (0, 0, 255), 2)

        cv2.putText(
            frame,
            img2char,
            (x1 + int(w1 / 50), y1 + int(h1 / 50)),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 0),
            2,
        )

    cv2.imshow("Webcam Text Detection", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
