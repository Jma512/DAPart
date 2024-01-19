import base64
import json
import os
import pickle
import sys
import time
import openpyxl as op
import torch
import pandas as pd
from PIL import Image
from flask import Flask, request
from io import BytesIO
from torchvision import transforms

from vgg16 import VGG16
from mobilenetv2 import MobileNetV2
from resnet50 import ResNet50

app = Flask(import_name=__name__)
app.config['SUBMODEL2'] = ''
app.config['SIZE'] = 224

class Util():

    def __init__(self):
        self.id = 0
        self.timelist = []

    def settimelist(self, time1, time2, time3):
        dict = {}
        dict["id"] = self.id
        dict["time1"] = time1
        dict["time2"] = time2
        dict["time3"] = time3
        self.id = self.id + 1
        self.timelist.append(dict)

    def gettimelist(self):
        return self.timelist

class Util2():

    def __init__(self):
        self.id = 0
        self.timelist = []

    def settimelist(self, time1, time2, time3, filesize):
        dict = {}
        dict["id"] = self.id
        dict["time1"] = time1
        dict["time2"] = time2
        dict["time3"] = time3
        dict["filesize"] = filesize
        self.id = self.id + 1
        self.timelist.append(dict)

    def gettimelist(self):
        return self.timelist

@app.route('/receivetestspeed', methods=['POST'])
def receive_data():
    data = request.data  # 获取POST请求的数据
    # 处理数据并返回结果
    return 'Received data: {}'.format(data)

@app.route('/startEdge', methods=["POST"])
def startEdge():
    submodel2 = app.config['SUBMODEL2']

    time1 = time.time()
    serialized_data = request.data
    size = sys.getsizeof(serialized_data)
    print('Serialized data size:', size, 'bytes')
    time6 = time.time()
    pre = pickle.loads(serialized_data)
    time3 = time.time()
    # print(pre.shape)
    # if pre.shape == (16):
    #     pre = pre.reshape((16, 128, 8, 8))
    # else:
    #     pre = pre.reshape((32, 128, 8, 8))
    pre = torch.tensor(pre, dtype=torch.float32).cuda()

    time4 = time.time()
    output = submodel2(pre).cpu()
    time5 = time.time()

    _, preds = torch.max(output, 1)
    preds = preds.detach().numpy()
    result = base64.b64encode(preds.tobytes())

    time2 = time.time()
    receive_time = time6 - time1
    computing_time = time2 - time1
    reference_time = time5 - time4
    util.settimelist(receive_time, reference_time, computing_time)
    # print(util.gettimelist())

    # print("serialized time: {}".format(time3 - time1))
    print("receive time: {}".format(receive_time))
    print("load time: {}".format(time3 - time6))
    print("server computing: {}".format(computing_time))
    print("reference time: {}".format(reference_time))

    print("发送成功")
    return result

@app.route('/startEdge2', methods=["POST"])
def startEdge2():
    submodel2 = app.config['SUBMODEL2']

    time1 = time.time()
    file_storage = request.files.get('file')
    # 将 FileStorage 对象转换为二进制数据
    file_data = file_storage.read()
    size = sys.getsizeof(file_data)
    print('Serialized data size:', size, 'bytes')
    # 将二进制数据转换为图像对象
    img = Image.open(BytesIO(file_data))
    time2 = time.time()
    img_name = img.filename
    if img is None:
        return "未上传文件"

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(app.config['SIZE']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        time3 = time.time()
        output = submodel2(img_tensor).cpu()

    _, preds = torch.max(output, 1)
    preds = preds.detach().numpy()

    result = label_dict[str(preds[0])][1]
    time4 = time.time()

    print("{} {} {}".format(i, result, img_name))

    receive_time = time2 - time1
    reference_time = time4 - time3
    computing_time = time4 - time1
    util.settimelist(receive_time, reference_time, computing_time)

    # print(util.gettimelist())
    # print("receive time: {}".format(receive_time))
    print("server reference time: {}".format(reference_time))
    # print("server computing: {}".format(computing_time))

    # print("发送成功")
    # app.config['SIZE'] = 224
    return result

def op_toExcel(data, fileName):  # openpyxl库储存数据到excel
    wb = op.Workbook()  # 创建工作簿对象
    ws = wb['Sheet']  # 创建子表
    ws.append(['序号', '接受数据时间', '处理时间', '服务器总时间'])  # 添加表头
    for i in range(len(data)):
        d = data[i]["id"], data[i]["time1"], data[i]["time2"], data[i]["time3"]
        ws.append(d)  # 每次写入一行
    wb.save(fileName)
    return "Server writes sucess!"

@app.route('/getposition', methods=["POST"])
def getposition():
    info = request.json
    print(info[0])
    print(info[1])
    if app.config['SUBMODEL2'] != '':
        del app.config['SUBMODEL2']
    submodel2 = get_modeltype(info[0], info[1])
    app.config['SUBMODEL2'] = submodel2

    return "success"

@app.route('/writeEXCEL', methods=["POST"])
def writeEXCEL():
    fileName = 'server_time_data.xlsx'
    response = op_toExcel(util.gettimelist(), fileName)
    print("服务器结果已保存")
    util.gettimelist().clear()
    util.id = 0
    util2.id = 0

    file = "local_time_data.xlsx"
    if not os.path.isfile(file):
        with open(file, 'w') as file:
            pass

    df1 = pd.read_excel('local_time_data.xlsx')
    df2 = pd.read_excel('server_time_data.xlsx')
    df2 = df2.drop(df2.columns[0], axis=1)
    merged_df = pd.concat([df1, df2], axis=1, ignore_index=False)
    merged_df.to_excel('result.xlsx', index=False)

    return response

@app.route('/writeEXCEL_neuro', methods=["POST"])
def writeEXCEL_neuro():
    fileName = 'server_time_data_neuro.xlsx'
    response = op_toExcel(util.gettimelist(), fileName)
    print("服务器结果已保存")
    util.gettimelist().clear()
    util.id = 0
    util2.id = 0

    file = "local_time_data.xlsx"
    if not os.path.isfile(file):
        with open(file, 'w') as file:
            pass

    df1 = pd.read_excel('local_time_data.xlsx')
    df2 = pd.read_excel('server_time_data.xlsx')
    df2 = df2.drop(df2.columns[0], axis=1)
    merged_df = pd.concat([df1, df2], axis=1, ignore_index=False)
    merged_df.to_excel('result.xlsx', index=False)

    return "ok"

@app.route('/getlocalexecl', methods=["POST"])
def getlocalexecl():
    execl = request.files.get('File')
    execl_name = execl.filename
    if execl is None:
        return "未上传文件"
    filepath = './' + execl_name
    execl.save(filepath)
    return "上传文件成功"

# @app.route('/test', methods=["POST"])
# def test():
#     execl = request.files.get('File')
#     execl_name = execl.filename
#     if execl is None:
#         return "未上传文件"
#     path = r"./001.jpg"  # 图片路径
#     img = cv.imdecode(np.fromfile("动漫人物_0.jpg",np.uint8))#含有中文路径的图片打开
#     img = cv2.imread(path)  # 读取图片
#     filepath = './' + execl_name
#     execl.save(filepath)
#     return "上传文件成功"

def get_modeltype(model, position):
    path = ""
    if model == "VGG16":
        path = "./model/vgg16/vgg16_pretrained_imagenet.pth"
        submodel2 = VGG16.get_split_presubvgg16_edge(path, position)
        app.config['SIZE'] = 224
    elif model == "MobileNetV2":
        path = "./model/mobilenetv2/mobilenetv2_pretrained.pth"
        submodel2 = MobileNetV2.get_split_presubmobilenetv2_edge(path, position)
        app.config['SIZE'] = 224
    elif model == "ResNet50":
        path = "./model/resnet50/resnet50_pretrained.pth"
        submodel2 = ResNet50.get_split_presubresnet50_edge(path, position)
        app.config['SIZE'] = 224
    # elif model == "AlexNet":
    #     path = "./model/alexnet/alexnet_pretrained.pth"
    #     submodel2 = AlexNet.get_split_presubalexnet_edge(path, position)
    #     app.config['SIZE'] = 224
    # elif model == "MobileViT":
    #     path = "./model/mobilevit/mobilenvit_pretrained.pth"
    #     submodel2 = MobileViT.get_split_presubmobilevit_edge(path, position)
    #     app.config['SIZE'] = 256
    else:
        print("No model type!")
        sys.exit(0)
    print(submodel2)
    submodel2.to(device)
    return submodel2

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(0))
    else:
        device = torch.device("cpu:0")
    print(device)

    label_path = "./data/imagenet_class_index.json"
    label_dict = json.load(open(label_path, "r"))
    i = 0
    util = Util()
    util2 = Util2()
    app.run(host='0.0.0.0', port=8022)