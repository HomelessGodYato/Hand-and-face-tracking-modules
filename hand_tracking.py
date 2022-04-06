import time

import cv2
import mediapipe as mp

previous_Time = 0
current_time = 0

width, height = 600, 400

capture = cv2.VideoCapture(0)
capture.set(3, width)
capture.set(4, height)

mp_draw = mp.solutions.drawing_utils

mHands = mp.solutions.hands
hands = mHands.Hands()

_, frame = capture.read()
h, w, c = frame.shape

while True:
    success, frame = capture.read()
    flip = cv2.flip(frame, 1)
    frame_RGB = cv2.cvtColor(flip, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_RGB)
    hand_lm = result.multi_hand_landmarks

    # print(result.multi_hand_landmarks)

    if hand_lm:
        for hand_landmarks in hand_lm:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
            cv2.rectangle(flip, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_draw.draw_landmarks(flip, hand_landmarks, mHands.HAND_CONNECTIONS)

    current_time = time.time()
    fps = 1 / (current_time - previous_Time)
    previous_Time = current_time

    cv2.putText(flip, "FPS: " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand tracker", flip)
    cv2.waitKey(1)
