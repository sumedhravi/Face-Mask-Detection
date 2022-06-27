# Face-Mask-Detection
Training an Image Segmentation model to identify whether people in the image are wearing a face mask and predicting the coordinates of the face mask. This implementation trains an object detection model using the YOLO v3 architecture and was tested with real world out of distribution images. Full training and prediction pipeline including data augmentation and hyperparameter tuning.

[YOLO](https://arxiv.org/abs/1506.02640), which stands for You Only Look Once, is a CNN-based object detection network. It provides real-time object detection performance by considering object detection as a regression problem. YOLO divides the image into grids and predicts bounding boxes, their confidence, and class probabilities simultaneously. Hence the training and evaluation time is significantly less than other alternative CNN based approaches. YOLOv2 and YOLOv3 are improved versions of the base YOLO network with several enhancements to improve detection accuracy. It uses a fully convolutional network with residual connections. Every convolutional layer is followed with a Batch Normalization layer and uses Leaky ReLU activation layer.

The base YOLO model often struggled with detections involving objects that were small due to downsampling of the input through the convolutional layers. [YOLO v3](https://arxiv.org/abs/1804.02767) adds residual blocks, skip connections and performs upsampling of the input which addresses these issues to an extent. YOLO v3 uses the Darknet-53 model as its backbone which contains 53 convolutional layers, trained on ImageNet. For the task of detection, 53 more layers are stacked onto it, giving us a 106 layer fully convolutional underlying architecture for YOLO v3. The bigger model makes the YOLO v3 version slower than the original YOLO model. The entire image is divided into an S x S grid and each grid cell generates B bounding boxes and confidence scores. Each bounding box prediction contains 4 attributes to define the bounding box coordinates and a confidence score indicating how confident the model is that an object is present in the bounding box. It also outputs the respective class probabilities. YOLO v3 makes prediction at three different layers, which helps address the issue of earlier versions of the YOLO model not detecting small objects.

## Dataset
The dataset used to train the model was a public dataset on [Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection). This dataset contains 863 images belonging to 3 different classes, wearing, not wearing and improperly wearing. The data was processed to merge the improperly wearing class with the not wearing class, and only 2 classes were used for training. Data augmentation was used to generate more training samples to avoid overfitting.

## Sample predictions
Sample predictions on test images.
![Sample prediction 1](/predictions/Yolo_predictions_1.jpg)


![Sample prediction 2](/predictions/yolo_predictions_2.jpg)
