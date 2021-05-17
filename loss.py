from model import DigitDetectionModelOutput, DigitDetectionModel
import torch 
import torch.nn.SmoothL1Loss
import torchvision.ops.sigmoid_focal_loss

class RetinaLoss:

    def compute_loss(
        self,
        model_output: DigitDetectionModelOutput,
        model_target: DigitDetectionModelTarget,
    ) -> Optional[torch.Tensor]: 
        model_target


class DigitAccuracy:

    def compute_metric(
        self,
        predicted_boxes: List[MnistBox],
        canvas: MnistCanvas,
    ):
        raise NotImplementedError