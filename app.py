# Pythonコード

from flask import Flask, render_template, request
import cv2
import mediapipe as mp
import numpy as np
import base64
import os
import json

app = Flask(__name__)

# zip関数をJinja2テンプレートに渡す
app.jinja_env.globals.update(zip=zip)

# MediaPipe Poseモデルの読み込み
mp_pose = mp.solutions.pose
poses = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# カスタムフィルターとしてb64encodeを定義
@app.template_filter('b64encode')
def b64encode_filter(s):
    return base64.b64encode(s.encode()).decode()

def manual_cos(A, B):
    if A is None or B is None:
        return 0.0  # もしくは適切な値に置き換えてください
    else:
        dot = np.sum(A*B, axis=-1)
        norm_A = np.linalg.norm(A, axis=-1)
        norm_B = np.linalg.norm(B, axis=-1)
        cos_similarity = dot / (norm_A * norm_B)
        return np.mean(cos_similarity)

def encode_image(image_dir, image_name):
    image_path = os.path.join(image_dir, image_name)
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

@app.route('/')
def index():
    return render_template('layout.html')

@app.route('/upload', methods=['POST'])
def compare_videos():
    # ファイルがアップロードされた場合
    if 'video1' in request.files and 'video2' in request.files:
        video1 = request.files['video1']
        video2 = request.files['video2']
        # 一時ディレクトリの作成
        os.makedirs("uploads", exist_ok=True)

        # アップロードされたファイルを一時ファイルとして保存
        video1_path = "uploads/video1.mp4"
        video2_path = "uploads/video2.mp4"
        video1.save(video1_path)
        video2.save(video2_path)

        # フレームを保存するディレクトリを作成
        video1_frames_dir = "uploads/video1_frames"
        os.makedirs(video1_frames_dir, exist_ok=True)

        # カメラやビデオファイルからの入力
        target_vid1 = cv2.VideoCapture(video1_path)
        target_vid2 = cv2.VideoCapture(video2_path)

        # スコア数のリスト
        totalscore = []

        frame_count = 0

        similarity_scores = []  # 類似度スコアを保持するリストを初期化する

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

            # フレームを保存
            video1_frame_path = os.path.join(video1_frames_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(video1_frame_path, img1)


            # コサイン類似度を計算
            score = manual_cos(pose_landmarks1, pose_landmarks2)
            totalscore.append(score)
            print("SCORE : " + str(score))
            # フレーム数をインクリメント
            frame_count += 1

        # スコアの平均を計算
        endScore = np.mean(totalscore)
        endScore = round(endScore * 100, 2)
        print("AVE SCORE : " + str(endScore))

        # 後処理
        cv2.destroyAllWindows()
        target_vid1.release()
        target_vid2.release()

        # 画像パスのリストをBase64エンコードされたデータのリストに変換
        encoded_images = [encode_image(video1_frames_dir, image_name) for image_name in os.listdir(video1_frames_dir)]


        # Cos類似度が低いフレームのインデックスを取得
        low_similarity_indices = np.argsort(totalscore)[:3]

        # スコアをフレーム数に合わせて増やす
        similarity_scores = [totalscore[i] for i in range(len(encoded_images))]
        similarity_scores = [round(score * 100, 2) for score in similarity_scores]

        # 結果をテンプレートに渡して表示
        return render_template('result.html', endScore=endScore, frames1=encoded_images, low_similarity_indices=low_similarity_indices, similarity_scores=similarity_scores)

    else:
        return "Error"

if __name__ == '__main__':
    app.run(debug=True)
