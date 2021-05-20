from model import *
from metric import *
import torch
import torchvision 
from torch.nn import SmoothL1Loss
from torchvision.ops import sigmoid_focal_loss 
from typing import List, Optional

class RetinaLoss:

    def compute_loss(
        self,
        model_output: DigitDetectionModelOutput,
        model_target: DigitDetectionModelTarget,
    ) -> Optional[torch.Tensor]: 
        loss_box_regression = 0
        loss_classification = 0 
        smooth = SmoothL1Loss()
        loss_box_regression += smooth(model_output.box_regression_output, model_target.box_regression_target)
        loss_classification += sigmoid_focal_loss(model_output.classification_output, model_target.classification_target, reduction='mean') 

        if len(model_target.matched_anchors) == 0:
          return None

        return (loss_box_regression + loss_classification)* model_output.classification_output.shape[1] / len(model_target.matched_anchors)


class DigitAccuracy:

    def compute_metric(
        self,
        predicted_boxes: List[MnistBox],
        canvas: MnistCanvas,
    ):
        print(len(canvas.boxes), 'canvas.boxes')
        print(len(predicted_boxes), 'predicted_boxes')
        for box in canvas.boxes:
          find = False
          for pbox in predicted_boxes:
            if box.iou_with(pbox) > 0.5 and box.class_nb == pbox.class_nb:
              if find: 
                print('tutaj wchodze')
                return 0
              find = True
          if not find:
            return 0
        
        if len(canvas.boxes) == len(predicted_boxes):
          return 1
        
        
        print('sprawdzam')

        return 0

              