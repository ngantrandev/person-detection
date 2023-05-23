import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=False,type=str, default="mobilenet_ssd/MobileNetSSD_deploy.prototxt",
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False,type=str, default="mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
    help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
    help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
    help="path to optional output video file")

ap.add_argument("-c", "--confidence", type=float, default=0.4, # confidence default 0.4
    help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=100,
    help="# of skip frames between detections")
args = vars(ap.parse_args())


# danh sách các class label mà mô hình hiện tại đã được train
LABELNAME = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]