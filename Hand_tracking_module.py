import time

import cv2
import mediapipe as mp
import numpy as np


class HandDetection:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mHands = mp.solutions.hands
        self.hands = self.mHands.Hands(static_image_mode=self.mode,
                                       max_num_hands=self.max_hands,
                                       min_detection_confidence=self.detection_confidence,
                                       min_tracking_confidence=self.track_confidence)

        self.mp_draw = mp.solutions.drawing_utils

    def hands_detection(self, frame, draw=True):
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_RGB)
        hand_lm = result.multi_hand_landmarks
        if hand_lm:
            for num, hand in enumerate(hand_lm):
                if draw:
                    x_min, y_min, x_max, y_max = self.calculate_box_coords(hand, frame)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    self.mp_draw.draw_landmarks(frame, hand, self.mHands.HAND_CONNECTIONS,
                                                self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                                         circle_radius=2),
                                                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2,
                                                                         circle_radius=2),
                                                )
                    if self.get_hand_label(num, hand, result):
                        text, coord = self.get_hand_label(num, hand, result)
                        cv2.putText(frame, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def get_hand_label(self, index, hand, results):
        output = None
        for idx, hand_detected in enumerate(results.multi_handedness):
            if hand_detected.classification[0].index == index:
                label = hand_detected.classification[0].label
                text = '{}'.format(label)

                # Extract Coordinates
                coords = tuple(np.multiply(
                    np.array((hand.landmark[self.mHands.HandLandmark.MIDDLE_FINGER_TIP].x,
                              hand.landmark[self.mHands.HandLandmark.MIDDLE_FINGER_TIP].y)),
                    [600, 400]).astype(int))

                output = text, coords
                print(label)

        return output

    def calculate_box_coords(self, hand, frame):
        h, w, c = frame.shape
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h
        for lm in hand.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            if x > x_max:
                x_max = x
            if y > y_max:
                y_max = y
            if x < x_min:
                x_min = x
            if y < y_min:
                y_min = y
        return x_min, y_min, x_max, y_max
