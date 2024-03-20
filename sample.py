import cv2
from os import makedirs
from os.path import splitext, dirname, basename, join

def save_frames(video_path: str, frame_dir: str, 
                name="image", ext="jpg"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    v_name = splitext(basename(video_path))[0]
    if frame_dir[-1:] == "//" or frame_dir[-1:] == "/":
        frame_dir = dirname(frame_dir)
    frame_dir_ = join(frame_dir, v_name)

    makedirs(frame_dir_, exist_ok=True)
    base_path = join(frame_dir_, name)

    idx = 0
    while cap.isOpened():
        idx += 1
        ret, frame = cap.read()
        if ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:  # 0秒のフレームを保存
                cv2.imwrite("{}_{}.{}".format(base_path, "0000", ext),
                            frame)
                cv2.imshow('Frame', frame)  # フレームを表示
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    break
            elif idx < cap.get(cv2.CAP_PROP_FPS)* 0.1:
                continue
            else:  # 0.5秒ずつフレームを保存
                second = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FPS))
                filled_second = str(second).zfill(4)
                cv2.imwrite("{}_{}.{}".format(base_path, filled_second, ext),
                            frame)
                cv2.imshow('Frame', frame)  # フレームを表示
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    break
                idx = 0
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
save_frames("./video/swing20230904.mp4", "./frame")