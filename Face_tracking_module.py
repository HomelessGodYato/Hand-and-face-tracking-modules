import time

import cv2
import mediapipe as mp
import numpy as np


class FaceDetection:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minConfidence=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minConfidence = minConfidence

        self.mFace = mp.solutions.face_mesh
        self.faceMesh = self.mFace.FaceMesh(static_image_mode=self.staticMode, max_num_faces=self.maxFaces,
                                            min_detection_confidence=self.minDetectionCon,
                                            min_tracking_confidence=self.minConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def face_detection(self, frame, draw=True):
        self.frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.faceMesh.process(self.frame_RGB)
        self.faceLandmarks = self.result.multi_face_landmarks
        if self.faceLandmarks:
            for face in self.faceLandmarks:
                self.mpDraw.draw_landmarks(frame, face, self.mFace.FACEMESH_CONTOURS,
                                           self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=1,
                                                                   circle_radius=1),
                                           self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2,
                                                                   circle_radius=2), )
        return frame


def main():
    curr_Time = 0
    prev_Time = 0
    cap = cv2.VideoCapture(0)
    face_detector = FaceDetection()
    while True:
        success, frame = cap.read()
        flip = cv2.flip(frame, 1)
        flip = face_detector.face_detection(flip)
        curr_Time = time.time()
        fps = 1 // (curr_Time - prev_Time)
        prev_Time = curr_Time
        cv2.putText(flip, "FPS: " + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Hand detection", flip)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
