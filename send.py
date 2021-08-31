import pandas as pd
import pprint
import argparse
import requests


def send(arg):
    DETECTION_URL = "http://192.168.219.203:5000/id-scan"
    TEST_IMAGE = arg.img

    image_data = open(TEST_IMAGE, "rb").read()

    response = requests.post(DETECTION_URL, files={"file_param_1": image_data}).json()
    # response2 = requests.post(DETECTION_URL, files={"file_param_face": image_data}).json()
    df = pd.DataFrame(response)
    # print(df)
    pprint.pprint(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str)
    args = parser.parse_args()
    send(args)