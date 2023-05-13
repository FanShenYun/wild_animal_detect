from flask import Flask, render_template, request, send_file
import supervision as sv
import ultralytics
from ultralytics import YOLO
from zipfile import ZipFile
from io import BytesIO
import torch
import numpy as np
import os
import csv

app = Flask(__name__)
model = YOLO('./best.pt')
CLASS_NAMES_DICT = model.model.names

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # 讀取上傳的影片
    video = request.files['video']
    video_path = os.path.join('static', video.filename)
    video_name = video.filename[:-4]
    video.save(video_path)
    TARGET_VIDEO_PATH = os.path.join('static',f"{video_name}_detect.mp4")
    video_info = sv.VideoInfo.from_video_path(video_path)
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    generator = sv.get_video_frames_generator(video_path)
    ani_result = {}
    with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for i,frame in enumerate(generator):
            temp = {}
            results = model(frame, conf=0.7, iou=0.5)
            detections = sv.Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )
            labels = [f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections]
            box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
            frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
            sink.write_frame(frame)
            for e in detections.class_id:
                if CLASS_NAMES_DICT[e] not in temp:
                    temp[CLASS_NAMES_DICT[e]] = 1
                else:
                    temp[CLASS_NAMES_DICT[e]] = temp[CLASS_NAMES_DICT[e]] + 1
            for i in temp:
                if i not in ani_result:
                    ani_result[i] = temp[i]
                elif i in ani_result and temp[i] > ani_result[i]:
                    ani_result[i] = temp[i]


    csv_path = os.path.join('static', f'{video_name}.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['species', 'number'])
        for i in ani_result:
            writer.writerow([i, ani_result[i]])
    torch.cuda.empty_cache()

    # 回傳下載連結
    return f'<a href="/download">下載結果</a>'

@app.route('/download')
def download():
    video_name = request.args.get('video')[:-11]
    video_path = os.path.join('static', request.args.get('video'))
    csv_path = os.path.join('static', request.args.get('csv'))
    stream = BytesIO()
    with ZipFile(stream, 'w') as zf:
        for file in [video_path,csv_path]:
            zf.write(file, os.path.basename(file))
    stream.seek(0)
    return send_file(stream, download_name=f'{video_name}.zip', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port='5000')