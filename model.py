import cv2
import mediapipe as mp

# MediaPipe Poseモデルの読み込み
mp_pose = mp.solutions.pose
poses = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) 

# カメラやビデオファイルからの入力
target_vid = cv2.VideoCapture("./video/swing20230904.mp4")

# 新しい動画サイズ
new_width = 700
new_height = 800

while True:
    success, img = target_vid.read()
    if not success:
        break

    img = cv2.resize(img, (new_width, new_height))

    # 画像をRGBに変換
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 姿勢推定を実行
    results = poses.process(imgRGB)

    # 姿勢ランドマークが検出された場合
    if results.pose_landmarks:
        # ランドマークを描画
        mp.solutions.drawing_utils.draw_landmarks(
            img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 結果を表示
    cv2.imshow('Frame', img)

    # 'q'キーが押されたら終了
    if cv2.waitKey(300) & 0xFF == ord('q'):
        break

# 後処理
cv2.destroyAllWindows()
target_vid.release()
