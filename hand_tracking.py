import cv2
import mediapipe as mp
import time

previous_Time = 0
current_time = 0

width, height = 600, 400

capture = cv2.VideoCapture(0)
capture.set(3, width)
capture.set(4, height)

mp_draw = mp.solutions.drawing_utils

mHands = mp.solutions.hands
hands = mHands.Hands()



while True:
    success, frame = capture.read()
    flip = cv2.flip(frame, 1)
    frame_RGB = cv2.cvtColor(flip, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_RGB)
    # print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(flip, hand_landmarks,mHands.HAND_CONNECTIONS)

    current_time = time.time()
    fps = 1/(current_time-previous_Time)
    previous_Time = current_time

    cv2.putText(flip, "FPS: " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand tracker", flip)
    cv2.waitKey(1)
