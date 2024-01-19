import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)
torch.save(model, "/home/root512/workspace/jetson_nano/model/vgg16/vgg16_pretrained_imagenet.pth")

print(model)
