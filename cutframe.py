import argparse
import subprocess

import cv2
import numpy as np
import tensorflow as tf


# ---------- Извлечение кадров (оба режима отдают RGB) ----------

def iter_frames_all(video_path):
    """Все кадры через OpenCV. Отдаём уже в RGB."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def iter_frames_ffmpeg(video_path):
    """Только ключевые кадры через ffmpeg, сразу в RGB (rgb24)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0:s=,", video_path],
        capture_output=True, text=True, check=True,
    )
    w_str, h_str = probe.stdout.strip().split(",")
    width, height = int(w_str), int(h_str)

    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-skip_frame", "nokey",
        "-i", video_path,
        "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", "rgb24",   # <-- сразу RGB
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_size = width * height * 3
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            # reshape без копирования — уже RGB, конвертировать не нужно
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


# ---------- Основной пайплайн ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="your_video.mp4")
    parser.add_argument("--model", default="my_classifier.keras")
    parser.add_argument(
        "--mode", choices=["all", "ffmpeg"], default="all",
        help="all — все кадры (OpenCV), ffmpeg — только ключевые кадры",
    )
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)

    frames = (
        iter_frames_all(args.video)
        if args.mode == "all"
        else iter_frames_ffmpeg(args.video)
    )

    votes_class_0 = 0
    votes_class_1 = 0
    frames_processed = 0
    tile_size = 224

    for frame_rgb in frames:                 # <-- уже RGB, cvtColor не нужен
        frames_processed += 1
        height, width, _ = frame_rgb.shape

        for y in range(0, height - tile_size + 1, tile_size):
            for x in range(0, width - tile_size + 1, tile_size):
                tile = frame_rgb[y:y + tile_size, x:x + tile_size]
                img_input = np.expand_dims(tile, axis=0)

                pred = model.predict(img_input, verbose=0)[0][0]
                if pred > 0.5:
                    votes_class_1 += 1
                else:
                    votes_class_0 += 1

    total_votes = votes_class_0 + votes_class_1

    print(f"Режим: {args.mode}, обработано кадров: {frames_processed}")

    if total_votes > 0:
        final_class = 1 if votes_class_1 > votes_class_0 else 0
        print(f"Всего обработано квадратов (голосов): {total_votes}")
        print(f"Голосов за Класс 0: {votes_class_0}")
        print(f"Голосов за Класс 1: {votes_class_1}")
        print(f"🏆 Итоговый класс видео: {final_class}")
    else:
        print("Видео оказалось слишком маленького разрешения, "
              "ни один квадрат 224x224 не поместился.")


if __name__ == "__main__":
    main()