from flask import Flask, render_template, request
import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# MediaPipe Poseモデルの読み込み
mp_pose = mp.solutions.pose
poses = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# 入力：比較する2つの座標ランドマーク群
# 出力：各ベクトルごとに比較したコサイン平均類似度
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
            np.delete(cos, count)
        count += 1
    return cos.mean()

@app.route('/')
def index():
    return render_template('layout.html')

@app.route('/upload', methods=['POST'])
def compare_videos():
    # ファイルがアップロードされた場合
    if 'swing20230506.mp4' in request.files and 'swing20230904.mp4' in request.files:
        video1 = request.files['swing20230506.mp4']
        video2 = request.files['swing20230904.mp4']

        # アップロードされたファイルを一時ファイルとして保存
        video1_path = "./uploads/video1.mp4"
        video2_path = "./uploads/video2.mp4"
        video1.save(video1_path)
        video2.save(video2_path)

        # カメラやビデオファイルからの入力
        target_vid1 = cv2.VideoCapture(video1_path)
        target_vid2 = cv2.VideoCapture(video2_path)

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

            # 画像をRGBに変換
            imgRGB1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            imgRGB2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            # 姿勢推定を実行
            results1 = poses.process(imgRGB1)
            results2 = poses.process(imgRGB2)

            # 姿勢ランドマークが検出された場合
            if results1.pose_landmarks:
                pose_landmarks1 = np.array([[lm.x, lm.y, lm.z] for lm in results1.pose_landmarks.landmark])
            else:
                pose_landmarks1 = None

            if results2.pose_landmarks:
                pose_landmarks2 = np.array([[lm.x, lm.y, lm.z] for lm in results2.pose_landmarks.landmark])
            else:
                pose_landmarks2 = None

            # 骨格描写を行い、描画された画像をリストに追加
            img1_with_pose = img1.copy()
            img2_with_pose = img2.copy()
            frames1.append(img1_with_pose)
            frames2.append(img2_with_pose)

            # コサイン類似度を計算
            score = manual_cos(pose_landmarks1, pose_landmarks2)
            totalscore.append(score)
            print("SCORE : " + str(score))

        # スコアの平均を計算
        endScore = np.mean(totalscore)
        print("AVE SCORE : " + str(endScore))

        # 後処理
        cv2.destroyAllWindows()
        target_vid1.release()
        target_vid2.release()

        # フレーム数の差を計算
        frame_diff = len(frames1) - len(frames2)

        # Cos類似度が低いフレームのインデックスを取得
        low_similarity_indices = np.argsort(totalscore)[:3]

        # 結果をテンプレートに渡して表示
        return render_template('result.html', endScore=endScore, frames1=frames1, low_similarity_indices=low_similarity_indices)

    else:
        return "Error: ファイルが提供されませんでした。"

if __name__ == '__main__':
    app.run(debug=True)
