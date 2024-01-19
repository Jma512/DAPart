import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
torch.save(model, "/home/root512/workspace/jetson_nano/model/resnet50/resnet50_pretrained_imagenet.pth")

# model = torch.load("/home/root512/workspace/jetson_nano/model/vit/vit_pretrained.pth")

print(model)
