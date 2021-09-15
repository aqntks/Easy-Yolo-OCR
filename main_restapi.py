
import io
import yaml
import torch
import argparse
from easyocr.easyocr import Reader

from core.scan import pt_detect
from yolov5.utils.torch_utils import time_sync
from yolov5.models.experimental import attempt_load

from PIL import Image
from flask import Flask, request
from waitress import serve

app = Flask(__name__)

DETECTION_URL = "/easy-yolo-ocr"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("file_param_1"):
        image_file = request.files["file_param_1"]
        image_bytes = image_file.read()

        img = Image.open(io.BytesIO(image_bytes))

        try:
            start_time = time_sync()
            pt_detect(img, device, detection_model, ciou, reader, gray=gray, byteMode=False)
            print('detecting time:', time_sync() - start_time)
        except Exception as e:
            print("detecting Fail")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--ciou', type=float, default=20)
    parser.add_argument('--gray', type=bool, default=False)
    parser.add_argument('--lang', type=str, default='en ko')
    args = parser.parse_args()

    gpu, gray, ciou, lang = args.gpu, args.gray, args.ciou, args.lang
    img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo', 'gif']

    # 디바이스 세팅
    if gpu == -1:
        dev = 'cpu'
    else:
        dev = f'cuda:{gpu}'
    device = torch.device(dev)

    # config 로드
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    img_path, detection = config['images'], config['detection']
    f.close()

    # 모델 세팅
    lang_list = lang.split(' ')
    reader = Reader(lang_list)

    detection_model = attempt_load(detection, map_location=device)

    print('----- 모델 로드 완료 -----')

    serve(app, host='0.0.0.0', port=args.port)
