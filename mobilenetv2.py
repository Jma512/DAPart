import json
import os
import time

import requests
import torch
from requests_toolbelt import MultipartEncoder
from torch import nn


def split_mobilenetv2(model):
    submodels = []
    current = nn.Sequential()
    for child in model.children():
        if isinstance(child, nn.Sequential):
            submodels.extend(split_mobilenetv2(child))
            # current.add_module(str(len(current)), nn.Flatten())
            # submodels.append(current)
        else:
            current.add_module(str(len(current)), child)
            # if len(current) == len(list(model.children())):
            submodels.append(current)
            current = nn.Sequential()

    return submodels


class MobileNetV2():

    @staticmethod
    def get_split_presubmobilebnetv2_mobile(path, info):
        model = torch.load(path)
        temp = split_mobilenetv2(model)
        current = nn.Sequential()
        current.add_module(str(len(current)), nn.AvgPool2d(kernel_size=7))
        current.add_module(str(len(current)), nn.Flatten())
        temp.insert(24, current)

        my_dict = {'0': 0, '1': 3, '2': 4, '3': 5, '4': 6, '5': 7, '6': 8, '7': 9, '8': 10, '9': 11, '10': 12, '11': 13,
                   '12': 14, '13': 15, '14': 16, '15': 17, '16': 18, '17': 19, '18': 20, '19': 24, '20': 25, '21':26}

        submodels = []
        length = len(my_dict)-1
        for i in range(length):
            current = nn.Sequential()
            start = my_dict[str(i)]
            end = my_dict[str(i + 1)]
            for j in range(start, end):
                current.add_module(str(len(current)), temp[j])
            submodels.append(current)

        submodel1 = nn.Sequential(*submodels[:info[1]])
        submodel1.load_state_dict(model.state_dict(), strict=False)

        return submodel1

    @staticmethod
    def edge_computing_mobilenetv2(label_path, data_path, util2, server_ip, server_port):
        label_dict = json.load(open(label_path, "r"))

        inverted_dict = {value[0]: key for key, value in label_dict.items()}

        # util2 = Util2()
        result = ""
        for filefolder in os.listdir(data_path):
            answer = inverted_dict[filefolder]
            temp = os.path.join(data_path, filefolder)
            for filename in os.listdir(temp):
                img_path = os.path.join(temp, filename)
                time1 = time.time()
                m = MultipartEncoder(
                    fields={'file': (filename, open(img_path, 'rb'), 'text/plain')})

                try:
                    response = requests.post(f'http://{server_ip}:{server_port}/startEdge2', data=m,
                                         headers={'Content-Type': m.content_type}, timeout=5)
                    print(response.text)
                    result = response.text
                except requests.exceptions.Timeout:
                    result = 0
                except requests.exceptions.RequestException as e:
                    result = 0

                # print(response.text)
                # result = response.text
                time2 = time.time()
                util2.settimelist(time2 - time1)
                total_time = time2 - time1

        try:
            response = requests.post(f'http://{server_ip}:{server_port}/writeEXCEL', data=[], timeout=3)
            # print(response.text)
        except requests.exceptions.Timeout:
            print("writeEXCEL timeout")
        except requests.exceptions.RequestException as e:
            print("writeEXCEL timeout")

        return util2, [result], total_time

    @staticmethod
    def get_split_presubmobilenetv2_edge(path, position):
        model = torch.load(path)
        temp = split_mobilenetv2(model)
        current = nn.Sequential()
        current.add_module(str(len(current)), nn.AvgPool2d(kernel_size=7))
        current.add_module(str(len(current)), nn.Flatten())
        temp.insert(24, current)

        my_dict = {'0': 0, '1': 3, '2': 4, '3': 5, '4': 6, '5': 7, '6': 8, '7': 9, '8': 10, '9': 11, '10': 12, '11': 13,
                   '12': 14, '13': 15, '14': 16, '15': 17, '16': 18, '17': 19, '18': 20, '19': 24, '20': 25, '21':26}

        submodels = []
        length = len(my_dict)-1
        for i in range(length):
            current = nn.Sequential()
            start = my_dict[str(i)]
            end = my_dict[str(i + 1)]
            for j in range(start, end):
                current.add_module(str(len(current)), temp[j])
            submodels.append(current)

        submodel2 = nn.Sequential(*submodels[position:])
        submodel2.load_state_dict(model.state_dict(), strict=False)

        return submodel2