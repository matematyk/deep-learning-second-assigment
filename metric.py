import torchvision
from typing import List
import torch
from model import MnistCanvas, MnistBox, DigitDetectionModelOutput

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
            classification_target[naj_ind, accept[i]['box'].class_nb] = 1
            box_regression_target[naj_ind, 0] = accept[i]['box'].x_min - anchors[i].x_min
            box_regression_target[naj_ind, 1] = accept[i]['box'].x_max - anchors[i].x_max
            box_regression_target[naj_ind, 2] = accept[i]['box'].y_min - anchors[i].y_min
            box_regression_target[naj_ind, 3] = accept[i]['box'].y_max - anchors[i].y_max


        return DigitDetectionModelTarget(classification_target, box_regression_target, matched_anchors)


                        

    def get_predictions(
        self,
        model_output: DigitDetectionModelOutput,
    ) -> List[MnistBox]:
        scores = torch.zeros(len(model_output.anchors))
        boxes = torch.zeros([len(model_output.anchors), 4])
        scores,_ = torch.max(model_output.classification_output[0], dim=1)
        for i in range(len(model_output.anchors)):
          boxes[i,0] = model_output.anchors[i].x_min - model_output.box_regression_output[0][i][0]
          boxes[i,1] = model_output.anchors[i].y_min - model_output.box_regression_output[0][i][2]
          boxes[i,2] = model_output.anchors[i].x_max - model_output.box_regression_output[0][i][1]
          boxes[i,3] = model_output.anchors[i].y_max - model_output.box_regression_output[0][i][3]

        indexes = torchvision.ops.nms(boxes, scores, 0.5)
        out_list = []
        for index in indexes:
          out_list.append(MnistBox(x_min=boxes[index,0],x_max=boxes[index,2],y_min=boxes[index,1], y_max=boxes[index,3], class_nb=torch.argmax(model_output.classification_output[0][index])))
  
        return out_list
