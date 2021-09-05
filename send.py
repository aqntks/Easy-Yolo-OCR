import pandas as pd
import pprint
import argparse
import requests


def send(arg):
    address = arg.address
    DETECTION_URL = f"http://{address}/easy-yolo-ocr"
    TEST_IMAGE = arg.img

    image_data = open(TEST_IMAGE, "rb").read()

    response = requests.post(DETECTION_URL, files={"file_param_1": image_data}).json()
    pprint.pprint(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str)
    parser.add_argument("--address", type=str, default='170.0.0.0:5000')
    args = parser.parse_args()
    send(args)