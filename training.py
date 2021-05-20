from canvas import MnistBox, MnistCanvas
from metric import TargetDecoder
from keras.datasets import mnist
import numpy as np
from typing import List, Optional
import torch
from loss import *
import torch.optim as optim
from model import DigitDetectionModel


mnist_data = mnist.load_data()
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist_data


def crop_insignificant_values(digit:np.ndarray, threshold=0.1):
    bool_digit = digit > threshold
    x_range = bool_digit.max(axis=0)
    y_range = bool_digit.max(axis=1)
    start_x = (x_range.cumsum() == 0).sum()
    end_x = (x_range[::-1].cumsum() == 0).sum()
    start_y = (y_range.cumsum() == 0).sum()
    end_y = (y_range[::-1].cumsum() == 0).sum()
    return digit[start_y:-end_y - 1, start_x:-end_x - 1]


TRAIN_DIGITS = [
    crop_insignificant_values(digit) / 255.0
    for digit_index, digit in enumerate(mnist_x_train[:10000])
]
TRAIN_CLASSES = mnist_y_train[:10000]

TEST_DIGITS = [
    crop_insignificant_values(digit) / 255.0
    for digit_index, digit in enumerate(mnist_x_test[:1000])
]
TEST_CLASSES = mnist_y_test[:1000]


def get_random_canvas(
    digits: Optional[List[np.ndarray]] = None,
    classes: Optional[List[int]] = None,
    nb_of_digits: Optional[int] = None,
    ):
    digits = digits if digits is not None else TRAIN_DIGITS
    classes = classes if classes is not None else TRAIN_CLASSES
    nb_of_digits = nb_of_digits if nb_of_digits is not None else np.random.randint(low=3, high=6 + 1)

    new_canvas = MnistCanvas.get_empty_of_size(size=(128, 128))
    attempts_done = 0
    while attempts_done < nb_of_digits:
        current_digit_index = np.random.randint(len(digits))
        current_digit = digits[current_digit_index]
        random_x_min = np.random.randint(0, 128 - current_digit.shape[0] - 3)
        random_y_min = np.random.randint(0, 128 - current_digit.shape[1] - 3)
        if new_canvas.add_digit(
            digit=current_digit,
            x_min=random_x_min,
            y_min=random_y_min,
            class_nb=classes[current_digit_index],
        ):
            attempts_done += 1
    return new_canvas


ANCHOR_SIZES = [16,19]

TEST_CANVAS_SIZE = 256
TEST_SEED = 42 # DO NOT CHANGE THIS LINE.

np.random.seed(TEST_SEED)

TEST_CANVAS = [
    get_random_canvas(
        digits=TEST_DIGITS,
        classes=TEST_CLASSES,
    )
    for _ in range(TEST_CANVAS_SIZE)
]



DEVICE = 'cpu'


import torch.optim as optim

model = DigitDetectionModel()
model.to(DEVICE)
print(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=2)
#@TODO, 
#zbieranie gradientu z wielu sampli
# liczba parametróœ 10k, 500mln 

rloss = RetinaLoss()
acc = DigitAccuracy()
target = TargetDecoder()

acc_add = 0
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params, 'total parameters')


#for epoch in range(1):
batch_size = 100
canvas = get_random_canvas()
for i in range(batch_size):

    optimizer.zero_grad()
    #canvas = get_random_canvas()
    outputs = model(canvas.get_torch_tensor())
    x = target.get_predictions(outputs)

    acc_value = acc.compute_metric(x, canvas)
    targets = target.get_targets(canvas, outputs.anchors, iou_threshold=0.5, nb_of_classes=10)
    if i == 20 or i == 40 or i == 60 or i == 99:
        canvas.plot(boxes=x, name=i)
    #print(x)
   # print(outputs.classification_output)
    #print(outputs.box_regression_output)

    loss = rloss.compute_loss(outputs, targets)

    print('treningowe accuracy:', acc_value)
    print('hello')
    print('loss', loss)
    loss.backward()
    optimizer.step()
    acc_add+=acc_value

    #acc_valid = 0
    #with torch.no_grad():
    #    for canvas in TEST_CANVAS:
    #        outputs = model(canvas.get_torch_tensor())
    #        acc_value = acc.compute_metric(target.get_predictions(outputs), canvas)
    #        loss = rloss.compute_loss(outputs, targets)
    #        print('valid_loss', loss)
    #        print('acc_loss', acc_value)
    #        acc_valid += acc_value

    #print('Accuracy of the network on the {} test images: {} %'.format(
    #    1, acc_valid))