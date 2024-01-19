import torch
import torchvision.models as models

model = models.mobilenet_v2(pretrained=True)
torch.save(model, "/home/root512/workspace/jetson_nano/model/mobilenetv2/mobilenetv2_pretrained_imagenet.pth")

print(model)
