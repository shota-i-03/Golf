#opencvとはopen source computer vision libraryで画像や動画の処理に必要な機能がまとめられたオープンソースのライブラリ
#pythonでopencvを利用するときは、import cv2

import cv2

#googleが提供している機械学習詰め合わせセット

import mediapipe as mp

#実行時間を測定するtimeをインポート
import time  

#opencvで画像取得するカメラを選択する
#カメラを1台接続している場合は、カメラのデバイスIDは0になる
#capはカメラのオブジェクト名となる

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

#無限ループを実行
while True:
    _, img = cap.read()   #opencvのカメラの画像取得コマンド
    #第1戻り値は画像の取得が成功したかどうかの結果がブール値で格納される
    #第2戻り値はカメラから取得した画像データが格納されている。格納されたデータは各画素毎にRGBの値を持っている3次元の配列データになっている
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #opencvはBGR,PillowはRGB
    #opencvの関数cvtColor()でBGRとRGBを変換
    results = hands.process(imgRGB)

    #もし手の検出があれば
    if results.multi_hand_landmarks: #手の各ランドマークの位置（出力はｘｙｚのリストで返ってくる）
        #検出されたそれぞれの手について繰り返す
        for hand_landmarks in results.multi_hand_landmarks:
            #手の各部位について繰り返す
            for i, lm in enumerate(hand_landmarks.landmark):
                #画像の高さ、幅、チャンネル数を取得
                height, width, channel = img.shape#インデックス番号、要素の順に取得できる
                #手の部位の座標を画像上の座標に変換
                cx, cy = int(lm.x * width), int(lm.y * height)
                #画像上の手の部位に番号を付ける
                cv2.putText(img, str(i+1), (cx+10, cy+10), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 5,cv2.LINE_AA)
                #画像上に手の部位を円で描画する
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            #手の部位同士を結ぶ線を描画する
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #画像を表示する
    cv2.imshow("Image", img)
    #'q'キーが押されるか、1ms待機し、何かキーが押されたかを確認してループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break