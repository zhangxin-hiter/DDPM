"""
predict.py
"""
import os.path

import torch
from torchvision import transforms
from PIL import Image
import json
from model.net import resnet, alexnet
import matplotlib.pyplot as plt

def main():

    image = Image.open("image.jpg")
    plt.imshow(image)

    data_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    print
    image = data_transform(image)

    image = torch.unsqueeze(image, dim=0)

    try:
        json_file = open("class_indices.json", "r")
        class_indices = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    net = alexnet.AlexNet(in_channels=3, out_features=5)

    model_weight_path = "checkpoints/net/alexnet.pth"
    if os.path.exists(model_weight_path):
        net.load_state_dict(torch.load(model_weight_path))
        print("load weight succesfully")

    net.eval()
    with torch.no_grad():
        output = torch.squeeze(net(image))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print(class_indices[str(predict_cla)], predict[predict_cla].item())
    plt.show()

if __name__ == "__main__":
    main()