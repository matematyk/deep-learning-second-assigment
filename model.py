import torch.nn as nn
from typing import List
from canvas import MnistBox, MnistCanvas
import torch


ANCHOR_SIZES = [16,19]

#backbone
class NetConv(nn.Module):
    def __init__(self):
        super(NetConv, self).__init__()
        modules = []

        modules.append(nn.Conv2d(1, 16, 3, padding=1))
        modules.append(nn.BatchNorm2d(16))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(2))

        modules.append(nn.Conv2d(16, 256, 3, padding=1))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(2))

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        x = self.layers(x)

        return x

import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.shape[0],) + self.shape)

class ClassificationHead(nn.Module):
    def __init__(self, anchors_numbers):
        super(ClassificationHead, self).__init__()
        modules = nn.ModuleList()
        #modules.append(nn.Flatten())
        #modules.append(nn.Linear(32*32*50, anchors_number*10))
        #(batch_size, in_channels= 256, 32, 32)
        per_pixel = 1
        modules.append(nn.Conv2d(in_channels=256, out_channels=(per_pixel*10), kernel_size=1))
        #(1, 961*10, 32, 32)
        #modules.append(Reshape(anchors_number,10))
        #modules.append(nn.Softmax(dim=2))


        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        x = self.layers(x)
        #(1, 961*10, 32, 32)
        x = x.permute(0,2,3,1)
        #(1, 32, 32, 961*10)
        x = x.reshape(-1, 10) 
        #(1*32*32*961, 10)
        return x

class BoxRegressionHead(nn.Module):
    def __init__(self, anchors_numbers):
        super(BoxRegressionHead, self).__init__()
        modules = nn.ModuleList()
        #modules.append(nn.Flatten())
        #modules.append(nn.Linear(32*32*50, anchors_numbers*4))
        per_pixel = 1
        modules.append(nn.Conv2d(in_channels=256, out_channels=per_pixel*4, kernel_size=1))
        #modules.append(Reshape(anchors_numbers, 4))
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        x = self.layers(x)
        #1, filters, 16, 16 = batch_size x liczba x wysoksoc x szeroksoc
        x = x.permute(0, 2, 3, 1)
        #batch_size x wysokoscx szerokosc x liczba_kanalow
        x = x.reshape(-1, 4) # [batch_size x wysoksoc x szeroksoc x liczba_anchorow(15)], 4

        return x

class DigitDetectionModelOutput:

    def __init__(
        self,
        anchors: List[MnistBox],
        classification_output: torch.Tensor,
        box_regression_output: torch.Tensor,
    ):
        self.anchors = anchors
        self.classification_output = classification_output
        self.box_regression_output = box_regression_output


class DigitDetectionModel(torch.nn.Module):
    # Should use ANCORS_SIZES
    anchors = []
    def __init__(
        self,
    ):
        super().__init__()
        self.netconv = NetConv()
        for n in range(0, 128//4):
          for m in range(0, 128//4):
              self.anchors.append(MnistBox(m*4 - ANCHOR_SIZES[0]/2, n*4 - ANCHOR_SIZES[1]/2, m*4 + ANCHOR_SIZES[1]/2, n*4+ANCHOR_SIZES[0]/2))

    def forward(self, x: MnistCanvas) -> DigitDetectionModelOutput:
        out = self.netconv(x)
        classification_target = ClassificationHead(len(self.anchors))(out)
        box_regression_output = BoxRegressionHead(len(self.anchors))(out)
        return DigitDetectionModelOutput(self.anchors, classification_target, box_regression_output)

