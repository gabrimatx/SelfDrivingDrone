import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import nn
from functools import partial
import numpy as np
import cv2

class ObstaclesDetector:
    """
    Class for obstacle detection. 
    """
    def __init__(self, model_path: str, model_type: str = "SSDLite") -> None:
        self._model_type = model_type
        self._model_path = model_path
        self._initialize_model()

    def detect_obstacles(self, frame: np.ndarray) -> np.ndarray:
        """
        Takes as an input a raw frame and predicts the position of every obstacle
        in the frame using the chosen model.
        """
        frame = self._transform(frame)
        if self._model_type == "Cascade":
            boxes = self._model.detectMultiScale(frame, 1.35, 30)
            boxes[:2] = boxes[:2] + boxes[2:] # [x0, y0, w, h] -> [x0, y0, x1, y1]
            return boxes
        
        else:
            frame = frame.to(self._device)
            with torch.no_grad():
                predictions = self._model([frame])
            
            boxes = predictions[0]['boxes'].cpu().numpy().astype(int)
            scores = predictions[0]['scores'].cpu().numpy()

            boxes = boxes[scores > 0.5]

            return boxes

    def draw_bounding_boxes(self, frame: np.ndarray, boxes: np.ndarray) -> None:
        """
        Takes as an input a raw frame and draws a box on every detected obstacle.
        """
        for box in boxes:
            self._draw_box(frame, box)

    def _draw_box(self, frame: np.ndarray, box: np.ndarray):
        """
        Helper method to draw a single box on a frame.
        """
        #box format: [x0, y0, x1, y1]
        up_left = box[:2]
        down_right = box[2:]
        cv2.rectangle(frame, up_left, down_right, (0, 0, 255), 2)

        center = (up_left + down_right) // 2
        cv2.circle(frame, center, 5, (0, 255, 0), cv2.FILLED)

    def _initialize_model(self):
        """
        Initializes the model chosen at constructor call.
        """
        if self._model_type == "Cascade":
            def transform(frame: np.ndarray):
                processed_frame = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))[0]
                processed_frame = cv2.GaussianBlur(frame, (5, 5), 0)
                return processed_frame
            
            self._transform = transform
            self._model = cv2.CascadeClassifier(self._model_path)
            return
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        num_classes = 2  # 1 class (obstacle) + background

        if self._model_type == "SSDLite":
            self._model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)

            in_channels = det_utils.retrieve_out_channels(self._model.backbone, (320, 320))
            num_anchors = self._model.anchor_generator.num_anchors_per_location()
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

            self._model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer)

        elif self._model_type == "Faster R-CNN":
            self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

            in_features = self._model.roi_heads.box_predictor.cls_score.in_features
            self._model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        else:
             print(self._model_type, type(self._model_type))
             raise ValueError(f"{self._model_type} in not a valid model. Parameter model type should be either Cascade, Faster-RCNN or SSDlite")
        
        self._model.load_state_dict(torch.load(self._model_path, map_location=self._device))
        self._model.eval()

        self._model = self._model.to(self._device)

        self._transform = T.Compose([T.ToTensor()])
