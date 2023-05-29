import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=False,type=str, default="mobilenet_ssd/MobileNetSSD_deploy.prototxt",
    help="đường dẫn đến file prototxt của model caffe")
ap.add_argument("-m", "--model", required=False,type=str, default="mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
    help="đường dẫn đến file model caffe")
ap.add_argument("--label", type=str, default="mobilenet_ssd/labels.names",
    help="đường dẫn đến file label của model")

ap.add_argument("-i", "--input", type=str,
    help="đường dẫn video đầu vào")

ap.add_argument("--stream", type=str, default="false",
    help="xác nhận có lấy video từ internet hoặc video có sẵn hay không")

ap.add_argument("-c", "--confidence", type=float, default=0.4,
    help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
    help="# of skip frames between detections")
args = vars(ap.parse_args())