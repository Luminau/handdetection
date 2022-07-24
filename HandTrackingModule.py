import time
import cv2
import mediapipe as mp


class HandDetector():
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.results = None
        self.p_time = 0
        self.c_time = 0
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode, self.max_num_hands, self.model_complexity,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hand(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            if draw:
                for handLms in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, handNo=0, draw=False):
        lm_list = []
        if self.results.multi_hand_landmarks:
            selected_hand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(selected_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, lm)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lm_list

    def draw_fps(self, img, draw=True):
        if draw:
            self.c_time = time.time()
            fps = 1 / (self.c_time - self.p_time)
            self.p_time = self.c_time
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
        return img


def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()

        img = detector.find_hand(img)
        img = detector.draw_fps(img)
        lm_list = detector.find_position(img)

        if len(lm_list) != 0:
            print(lm_list[4])

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
