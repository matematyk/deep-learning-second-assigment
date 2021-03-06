import torchvision
from typing import List
import torch
from model import MnistCanvas, MnistBox, DigitDetectionModelOutput
import torch.nn as nn

class DigitDetectionModelTarget:

    def __init__(
        self,
        classification_target: torch.Tensor,
        box_regression_target: torch.Tensor,
        matched_anchors: List[int],
    ):
        self.classification_target = classification_target
        self.box_regression_target = box_regression_target
        self.matched_anchors = matched_anchors


class TargetDecoder:

    def get_targets(
        self,
        canvas: MnistCanvas,
        anchors: List[MnistBox],
        iou_threshold: float=0.5,
        nb_of_classes: int = 10,
    ) -> DigitDetectionModelTarget:
        matched_anchors = []
        classification_target = torch.zeros([len(anchors), nb_of_classes])
        box_regression_target = torch.zeros([len(anchors), 4])
        for i in (range(len(anchors))):
          accept = []
          for box_a in canvas.boxes:
            if box_a.iou_with(anchors[i]) >= iou_threshold:
              accept.append({'box':box_a, 'iou': box_a.iou_with(anchors[i])})
              matched_anchors.append(i)
          if len(accept) > 0:
            naj_ind = 0
            for j in range(len(accept)):
                if accept[j]['iou'] > accept[naj_ind]['iou']:
                  naj_ind = j
            classification_target[i, accept[naj_ind]['box'].class_nb] = 1
            box_regression_target[i, 0] = accept[naj_ind]['box'].x_min - anchors[i].x_min
            box_regression_target[i, 1] = accept[naj_ind]['box'].x_max - anchors[i].x_max
            box_regression_target[i, 2] = accept[naj_ind]['box'].y_min - anchors[i].y_min
            box_regression_target[i, 3] = accept[naj_ind]['box'].y_max - anchors[i].y_max


        return DigitDetectionModelTarget(classification_target, box_regression_target, matched_anchors)


    def get_predictions(
        self,
        model_output: DigitDetectionModelOutput,
    ) -> List[MnistBox]:
        scores = torch.zeros(len(model_output.anchors))
        boxes = torch.zeros([len(model_output.anchors), 4])

        scores,_ = torch.max(model_output.classification_output, dim=1)
    
        for i in range(len(model_output.anchors)):
          boxes[i,0] = model_output.anchors[i].x_min - model_output.box_regression_output[i][0]
          boxes[i,1] = model_output.anchors[i].y_min - model_output.box_regression_output[i][2]
          boxes[i,2] = model_output.anchors[i].x_max - model_output.box_regression_output[i][1]
          boxes[i,3] = model_output.anchors[i].y_max - model_output.box_regression_output[i][3]
        indexes = torchvision.ops.nms(boxes, scores, 0.1)
        out_list = []

        #todo zrobic 3 najwieksze i + 3 boxex        
        m = nn.Softmax(dim=1)
        #64x10
        prob = m(model_output.classification_output[indexes])
        values_for_boxes, ind= torch.max(prob,dim=1)
        print(values_for_boxes)
        lista = list(zip(values_for_boxes,indexes))
        #64
        sorted_by_first = sorted(lista, key=lambda tup: tup[0], reverse=True)
        indexes = [x[1] for x in sorted_by_first[:3] if x[0] > 0.02]
        
        for index in indexes:
          out_list.append(MnistBox(x_min=boxes[index, 0], x_max=boxes[index, 2], y_min=boxes[index, 1], y_max=boxes[index,3], class_nb=torch.argmax(model_output.classification_output[index])))
        
        return out_list
