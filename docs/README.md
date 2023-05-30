# SelfDrivingDrone

SelfDrivingDrone is an AI software system that implements state-of-the-art object detection algorithms, including [Faster R-CNN](https://arxiv.org/abs/1506.01497) and [SSDlite](https://arxiv.org/abs/1512.02325), to fly a drone through an obstacle course. It is written in Python and powered by the [djitellopy](https://github.com/damiafuentes/DJITelloPy) and [simple-pid](https://pypi.org/project/simple-pid/) libraries.

<div align="center">
  <img src="readme_figs/demo.gif" width="480px" />
  <p>Example of SelfDrivingDrone execution. Watch the complete video <a href="https://www.youtube.com/watch?v=cFo53diY-kI&ab_channel=RubenCiranni">here</a>.</p>
</div>

# Introduction

The goal of SelfDrivingDrone is to provide a high-quality, high-performance code for all those who want to fly their Tello drones through any obstacle course.

SelfDrivingDrone exploits fine-tuned implementations of the following object detection algorithms:

- [Faster R-CNN](https://arxiv.org/abs/1506.01497): Slow, but extremely precise. 
- [SSDlite](https://arxiv.org/abs/1512.02325): Fast, but less precise than Faster R-CNN. Best option for users lacking a dedicated graphics card.
- [Haar Cascade Classifier](https://ieeexplore.ieee.org/document/990517): Fast, but less precise than SSDlite.

<br>
<br>

# Installation

**Requirements:**

- Python3, Dji Tello drone

<div align="center">
  <img src=".\readme_figs\drone-dji-tello.jpg" width="300px" />
  <p>An example of a Dji Tello drone.</p>
</div>

<br>

### **Clone the SelfDrivingDrone repository:**

```
# SELF_DRIVING_DRONE=/path/to/clone/SelfDrivingDrone
git clone https://github.com/gabrimatx/SelfDrivingDrone $SELF_DRIVING_DRONE
```

### **Install Python dependencies:**

```
pip install -r $SELF_DRIVING_DRONE/requirements.txt
```

- [PyTorch](https://pytorch.org/get-started/locally/)

<br>
<br>

# Quick Start: Using SelfDrivingDrone

During installation, you will be provided with 3 pre-trained models designed for Object Recognition on the obstacle depicted below. Additionally, you will receive a way to run them automatically through the `main.py` file.

<div align="center">
  <img src=".\readme_figs\obstacle.jpeg" width="300px" />
  <p>Obstacle example.</p>
</div>

Please, note that any obstacles of similar shape will be suitable to be recognized by our models, but the color of the paper may have a small impact on the results.

<br>

### **How to run the project:**

Choose one of the following lines based on the model of your choice:

```
# SELF_DRIVING_DRONE=/path/to/clone/SelfDrivingDrone
python main.py cascade 
python main.py ssdlite 
python main.py faster-rcnn 
$SELF_DRIVING_DRONE
```

 - If no parameter is passed through the command line to the `main.py` file, the default executed model will be the SSDlite model. 

- the default executed model can be changed at will in the `main.py` file, simply by changing the value of the `default_model` variable among `Cascade`, `SSDlite` and `faster-RCNN`. (the assignments are case insensitive)

<br>

### **Visualization**
As a finishing touch, 2 real time windows are initiated, each operating on the same dedicated thread. These windows serve the purpose of visualizing the following: 
1. The live video feed captured by the drone, with bounding boxes drawn around any obstacles present in the scene.
2. A graphical representation, comprising 3 `matplotlib.subplots`, that illustrate the displacement of the nearest obstacle’s center (divided into its x,y components)
relative to the setpoints for each PID along with the area of the
obstacle. The Area is given by the number of pixels it occupies within the scene.

<br>
<br>

# Authors

We're a group of three hard-working university students at Sapienza University of Rome. SelfDrivingDrone is a project that we have undertaken as part of our academic curriculum, specifically designed to fulfill the requirements of one of our examinations -  AI Lab: Computer Vision and NLP.

The entire process has been nothing short of stimulating and fulfilling, and we sincerely hope that our passion and dedication shine through in the final outcome. For any clarifications or further information regarding the project, please feel free to reach out to us.


<img src=".\readme_figs\AUTHORS.svg">

<br>
<br>

## License

SelfDrivingDrone is released under the [MIT License](./LICENSE). 
