import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 300)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


while True:
    success, frame = cap.read()
    if success:
        RGB_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(RGB_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                print(hand_landmarks)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
        cv.imshow("Gesture Tracking", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
