import time

import cv2

import Face_tracking_module as ftm
import Hand_tracking_module as htm


def main():
    curr_Time = 0
    prev_Time = 0
    cap = cv2.VideoCapture(0)
    hand_detector = htm.HandDetection()
    face_detector = ftm.FaceDetection()
    while True:
        success, frame = cap.read()
        flip = cv2.flip(frame, 1)
        flip = hand_detector.hands_detection(flip)
        flip = face_detector.face_detection(flip)
        curr_Time = time.time()
        fps = 1 // (curr_Time - prev_Time)
        prev_Time = curr_Time
        cv2.putText(flip, "FPS: " + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Hand detection", flip)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
