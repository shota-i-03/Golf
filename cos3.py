import cv2
import mediapipe as mp
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

# MediaPipe Poseモデルの読み込み
mp_pose = mp.solutions.pose
poses = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

#入力：比較する2つの座標ランドマーク群
#出力：各ベクトルごとに比較したコサイン平均類似度

def manual_cos(A, B):
    dot = np.sum(A*B, axis=-1)
    A_norm = np.linalg.norm(A, axis=-1)
    B_norm = np.linalg.norm(B, axis=-1)
    cos = dot / (A_norm*B_norm+1e-10)

    # 検出できない場合の処理
    for i in cos:
        count = 0
        if i == 0:
            print("cos deleted")
            np.delete(cos,count)
        count = count +1
    return cos.mean()


# カメラやビデオファイルからの入力
target_vid1 = cv2.VideoCapture("./video/swing20230904.mp4")
target_vid2 = cv2.VideoCapture("./video/swing20230506.mp4")

# 新しい動画サイズ
new_width = 700
new_height = 800

# フレーム数のカウント
frame_count = 0

# 前のフレームのポーズ
prev_pose_landmarks1 = None
prev_pose_landmarks2 = None

# 同じポーズを持つフレームのカウント
same_pose_frame_count1 = 0
same_pose_frame_count2 = 0

# フレームを格納するリスト
frames1 = []
frames2 = []

# スコア数のリスト
totalscore = []

while True:
    success1, img1 = target_vid1.read()
    success2, img2 = target_vid2.read()

    # どちらかの動画が終了した場合、ループを抜ける
    if not success1 or not success2:
        break


    # リサイズ
    img1 = cv2.resize(img1, (new_width, new_height))
    img2 = cv2.resize(img2, (new_width, new_height))

    # 画像をRGBに変換
    imgRGB1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    imgRGB2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 姿勢推定を実行
    results1 = poses.process(imgRGB1)
    results2 = poses.process(imgRGB2)

    # 姿勢ランドマークが検出された場合
    if results1.pose_landmarks:
        # ランドマークを描画
        mp.solutions.drawing_utils.draw_landmarks(
            img1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        pose_landmarks1 = np.array([[lm.x, lm.y, lm.z] for lm in results1.pose_landmarks.landmark])
    else:
        pose_landmarks1 = None

    if results2.pose_landmarks:
        # ランドマークを描画
        mp.solutions.drawing_utils.draw_landmarks(
            img2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        pose_landmarks2 = np.array([[lm.x, lm.y, lm.z] for lm in results2.pose_landmarks.landmark])
    else:
        pose_landmarks2 = None

    # 前のフレームのポーズを更新
    prev_pose_landmarks1 = pose_landmarks1
    prev_pose_landmarks2 = pose_landmarks2

    # 同じポーズが3枚以上の場合、フレームを1枚にする    
    if same_pose_frame_count1 >= 3:
        frames1 = [frames1[-1]]  # 最新のフレームを残す
        same_pose_frame_count1 = 0  # 同じポーズのフレーム数をリセット

    if same_pose_frame_count2 >= 3:
        frames2 = [frames2[-1]]  # 最新のフレームを残す
        same_pose_frame_count2 = 0  # 同じポーズのフレーム数をリセット

    # フレーム数の差を取得
    frame_diff = len(frames1) - len(frames2)

    # フレーム数が異なる場合
    if frame_diff != 0:
        if frame_diff > 0:  # frames1の方が多い場合
        # frames1から削るフレーム数を計算
            frames_to_remove = frame_diff
        # フレームを等間隔に削る
            step = len(frames1) // frames_to_remove
            frames1 = [frames1[i] for i in range(len(frames1)) if i % step != 0]
        else:  # frames2の方が多い場合
        # frames2から削るフレーム数を計算
            frames_to_remove = abs(frame_diff)
        # フレームを等間隔に削る
            step = len(frames2) // frames_to_remove
            frames2 = [frames2[i] for i in range(len(frames2)) if i % step != 0]


        # コサイン類似度を計算
        score = manual_cos(pose_landmarks1, pose_landmarks2)
        totalscore.append(score)
        print("SCORE : " + str(score))

    frame_count += 1

# 同じフレーム数になったか確認
print("Video 1 Frame Count:", len(frames1))
print("Video 2 Frame Count:", len(frames2))

endScore = np.mean(totalscore)
print("AVE SCORE : " + str(endScore))

# フレームをコマ送りで表示
for frame1, frame2 in zip(frames1, frames2):
    cv2.imshow('Frame 1 with Pose', frame1)
    cv2.imshow('Frame 2 with Pose', frame2)

    # 'q'キーが押されたら終了
    if cv2.waitKey(300) & 0xFF == ord('q'):
        break

# 後処理
cv2.destroyAllWindows()
target_vid1.release()
target_vid2.release()
