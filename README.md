# Custom Object Detection using YOLOv8 for Airplane Detection from Aerial Images

![image](https://github.com/Shivapriya1726/ObjectDetection_using_Yolov8/assets/90460346/51cf0ee0-a2a8-4bf2-a152-0e1998488a27)

## Project Overview
Welcome to the repository for the "Custom Object Detection using YOLOv8" project, focused on the detection of airplanes from aerial images. This project utilizes the YOLOv8 architecture to perform accurate and efficient object detection, specifically targeting airplanes within aerial views.

## About YOLOv8
YOLO (You Only Look Once) is a state-of-the-art object detection algorithm that achieves remarkable accuracy and speed in real-time object detection tasks. YOLOv8 is an evolution of the YOLO series, combining deep learning techniques to enhance object detection capabilities, making it an ideal choice for detecting airplanes in aerial images.

## Project Features
1. Aerial Object Detection: Our project specializes in detecting airplanes in aerial images, a task of significant importance in fields like surveillance, security, and urban planning.
2. YOLOv8 Implementation: We have implemented the YOLOv8 algorithm, which provides a robust framework for real-time object detection with impressive accuracy.
3. Web Application: Explore our interactive web application that showcases the capabilities of our YOLOv8 model in detecting airplanes. Visit the application [here](https://objectdetectionusingyolov8-apsfg7eodidevxqhaqtey6.streamlit.app/).

## How to Use the Application
1. Access the web application using the provided link: [Airplane Detection App](https://objectdetectionusingyolov8-apsfg7eodidevxqhaqtey6.streamlit.app/).
2. Upload an aerial image that you want to analyze for airplane detection.
3. Initiate the detection process by clicking the relevant button on the web interface.
4. Wait for the algorithm to process the image. Detected airplanes will be highlighted with bounding boxes.
5. View the results on the processed image and also the number of airplanes detected on the image.

## Data collection
Data of aerial view images of airplanes are collected from the Roboflow : [data](https://universe.roboflow.com/lewis-ly9ii/aerial-imagery-xbdqz)
The dataset includes 3176 images.
The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Randomly crop between 0 and 40 percent of the image

## Training Process
Training an object detection model like YOLOv8 involves several steps to achieve accurate and reliable results. Here's an overview of the training process for your Custom Object Detection project focused on airplane detection from aerial images:
1. Model Selection: YOLOv8m, a variant of the YOLOv8 architecture, was chosen as the base model for this project. YOLOv8m combines deep learning techniques to improve object detection accuracy while maintaining real-time detection capabilities.
2. Training Environment: The training process was carried out in Google Colab, utilizing its GPU capabilities for faster model convergence. Colab provides a cloud-based environment that supports deep learning tasks, enabling the training of complex models like YOLOv8m.
3. Epochs and Time: The model was trained for 15 epochs, with each epoch representing a full pass through the entire dataset. Training YOLOv8m for 15 epochs took approximately 10 hours. Longer training times are often necessary to fine-tune the model's weights and optimize its performance.

## Results
![image](https://github.com/Shivapriya1726/ObjectDetection_using_Yolov8/assets/90460346/ad374bad-32fe-4c7c-a901-9d3dfa16dded)   ![image](https://github.com/Shivapriya1726/ObjectDetection_using_Yolov8/assets/90460346/c241f99f-c9ff-46d4-bc64-56b3db40e8f8)

1. Model Architecture: The model utilized for validation is YOLOv8, with a fused architecture containing 168 layers and a total of 3,005,843 parameters.
2. Validation Metrics:
  * Class Accuracy: The model achieved an overall class accuracy of 0.908. This metric indicates the proportion of correctly predicted classes in the validation dataset.
  * Recall (R@0.861): The recall at a confidence threshold of 0.861 was calculated as 0.915. Recall measures the proportion of true positive instances captured by the model.
  * mAP50: The mean average precision at IoU threshold 0.50 (mAP50) was recorded as 0.491. mAP measures the model's accuracy in predicting bounding box positions.
  * mAP50-95: The model achieved 100% mAP at IoU thresholds ranging from 0.50 to 0.95, indicating strong performance across various levels of precision.


