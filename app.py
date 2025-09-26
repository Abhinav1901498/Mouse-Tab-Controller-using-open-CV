import cv2
import mediapipe as mp
import pyautogui
import time
import platform

# --------- Mediapipe Hands Setup ---------
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# --------- Webcam Setup ---------
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()  # स्क्रीन का resolution

# --------- Gesture Control Variables ---------
prev_x, prev_y = 0, 0
smoothening = 5
last_action_time = 0
action_delay = 1   # हर gesture के बीच gap

# --------- Finger Tips IDs ---------
tipIds = [4, 8, 12, 16, 20]

# --------- Finger Count Function ---------
def count_fingers(lmList):
    fingers = 0
    if lmList[tipIds[0]][0] > lmList[tipIds[0]-1][0]:
        fingers += 1
    for id in range(1, 5):
        if lmList[tipIds[id]][1] < lmList[tipIds[id]-2][1]:
            fingers += 1
    return fingers

# --------- Main Loop ---------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            if lmList:
                fingers = count_fingers(lmList)

                # Mouse Move (Index finger tip)
                ix, iy = lmList[8]
                screen_x = screen_width * ix / w
                screen_y = screen_height * iy / h
                curr_x = prev_x + (screen_x - prev_x) / smoothening
                curr_y = prev_y + (screen_y - prev_y) / smoothening
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # Action delay
                current_time = time.time()

                # 👊 Fist → Mouse Click
                if fingers == 0 and current_time - last_action_time > action_delay:
                    pyautogui.click()
                    last_action_time = current_time

                # ✌️ 2 Fingers → Close Current Tab
                elif fingers == 2 and current_time - last_action_time > action_delay:
                    if platform.system() == "Darwin":  # MacOS
                        pyautogui.hotkey("command", "w")
                    else:  # Windows/Linux
                        pyautogui.hotkey("ctrl", "w")
                    last_action_time = current_time

                # 🖐️ 5 Fingers → Close Entire Window
                elif fingers == 5 and current_time - last_action_time > action_delay:
                    if platform.system() == "Darwin":
                        pyautogui.hotkey("command", "q")
                    else:
                        pyautogui.hotkey("alt", "f4")
                    last_action_time = current_time

                # Show finger count on screen
                cv2.putText(frame, f'Fingers: {fingers}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
