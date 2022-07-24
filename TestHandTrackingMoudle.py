import cv2
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
detector = htm.HandDetector()
while True:
    success, img = cap.read()

    img = detector.find_hand(img)
    img = detector.draw_fps(img)
    lm_list = detector.find_position(img)

    if len(lm_list) != 0:
        print(lm_list[8])

    cv2.imshow("Image", img)
    cv2.waitKey(1)

