import cv2
import mediapipe as mp
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

def landmark2np(hand_landmarks):
    li = []
    for j in (hand_landmarks.landmark):
        li.append([j.x, j.y, j.z])

    return np.array(li) - li[0]


def manual_cos(A, B):
    dot = np.sum(A*B, axis=-1)
    A_norm = np.linalg.norm(A, axis=-1)
    B_norm = np.linalg.norm(B, axis=-1)
    cos = dot / (A_norm*B_norm+1e-7)
    print(cos[1:].mean())

    return cos[1:].mean()


cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

saved_array = [None, None, None]
start = -100
score = [0, 0, 0]
saved_no = 0

while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i, lm in enumerate(hand_landmarks.landmark):
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.putText(img, str(i+1), (cx+10, cy+10), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)


            if cv2.waitKey(1) & 0xFF == ord('s'):
                saved_array[0] = landmark2np(hand_landmarks)
                start = time.time()
                saved_no = 1
                print('no.1 saved')
            
            if cv2.waitKey(1) & 0xFF == ord('d'):
                saved_array[1] = landmark2np(hand_landmarks)
                start = time.time()
                saved_no = 2
                print('no.2 saved')
            
            if cv2.waitKey(1) & 0xFF == ord('f'):
                saved_array[2] = landmark2np(hand_landmarks)
                start = time.time()
                saved_no = 3
                print('no.3 saved')

            # cos類似度でチェック
            if saved_array[0] is not None:
                now_array = landmark2np(hand_landmarks)
                score[0] = manual_cos(saved_array[0], now_array)

            if saved_array[1] is not None:
                now_array = landmark2np(hand_landmarks)
                score[1] = manual_cos(saved_array[1], now_array)

            if saved_array[2] is not None:
                now_array = landmark2np(hand_landmarks)
                score[2] = manual_cos(saved_array[2], now_array)

    # 3s 表示
    if time.time() - start < 3:
        cv2.putText(img, f'No.{saved_no} saved', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), thickness=2)

    elif score[0] > 0.99:
        cv2.putText(img, 'no.1 pose', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 255), thickness=2)

    elif score[1] > 0.99:
        cv2.putText(img, 'no.2 pose', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 255), thickness=2)

    elif score[2] > 0.99:
        cv2.putText(img, 'no.3 pose', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 255), thickness=2)
        

    cv2.imshow("Image", img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break