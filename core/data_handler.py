import os
import json
from collections import OrderedDict
from core.image_handler import *


def masking_save(image_pack, result, path):
    filename = path.split('/')[-1].split('.')[0]
    img = image_pack.getOImg()

    for rect in result:
        x1, y1, x2, y2 = rect['x1'], rect['y1'], rect['x2'], rect['y2']
        img[y1:y2, x1:x2] = 0

    cv2.imwrite(f'data/masking/{filename}.jpg', img)


def make_json_masking(masking_result, easy_all_result, img_path, finger_count):
    finish = OrderedDict()
    finish['count'] = len(masking_result) - finger_count

    if masking_result or easy_all_result:
        finish['masking'] = masking_result
        finish['ocr'] = easy_all_result

    savePath = img_path.replace('link', 'json')
    folder = savePath[:savePath.rfind('/')]
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(f'{savePath}.json', "w") as json_file:
        json.dump(finish, json_file, ensure_ascii=False, indent='\t')


def printResult_passport(fileName, passport):
    if passport is None:
        print(print(f"\n\n----- {fileName} 검출 실패 -----"))
        return

    passportType, issuingCounty, sur, given, passportNo, nationality, birth, sex, expiry, personalNo = passport.all()

    # result print
    print(f"\n\n----- {fileName} Passport Scan Result -----")
    print('Type            :', passportType)
    print('Issuing county  :', issuingCounty)
    print('Passport No.    :', passportNo)
    print('Surname         :', sur)
    print('Given names     :', given)
    print('Nationality     :', nationality)
    # print('Personal No.    :', personalNo)
    print('Date of birth   :', birth)
    print('Sex             :', sex)
    print('Date of expiry  :', expiry)
    print("---------------------------------------\n")