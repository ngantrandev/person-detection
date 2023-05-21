import cv2
import argparse
import imutils
from vidgear.gears import CamGear
from mylib.centroidtracker import CentroidTracker
import dlib
import numpy as np
from imutils.video import FPS

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
ap.add_argument("-s", "--skip-frames", type=int, default=10,
    help="# of skip frames between detections")
args = vars(ap.parse_args())

# danh sách các class label mà mô hình hiện tại đã được train
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

def main():
    totalFrames = 0
    # lưu trữ kích thước khung hình, lấy từ frame đầu tiên
    H = 0
    W = 0

    # read cafe model from mobilenet_ssd folder using args
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # list of trackers for tracking process
    trackers = []
    # tạo đối tượng dùng để tracking
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)

    #khởi tạo đối tượng FPS để đo tốc độ xử lý
    fps = FPS().start()
    vd = CamGear(source = "https://youtu.be/d0mcaxedv_4", stream_mode = True).start()


    # vd = cv2.VideoCapture("https://youtu.be/T2-3J2hWjbk")
    while True:
        frame = vd.read()
        # thay đổi kích thước khung hình thành 500px. càng ít dữ liệu thì xử lý càng nhanh
        frame = imutils.resize(frame, width = 500)
        # chuyển đổi khung hình sang RGB
        #vì dlib cần RGB để xử lý
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # lấy kích thước khung hình từ frame đầu tiên
        if W == 0 or H == 0:
            (H, W) = frame.shape[:2]

        # nhận diện vật thể
        if totalFrames % args["skip_frames"] == 0:
            trackers = [] # luư thông tin các đối tượng đã tracking

            # tạo blob từ frame để truyền vào Net để nhận diện object
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # lặp qua các detections
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                # lọc các detections có confidence lớn hơn confidence nhỏ nhất
                if confidence > args["confidence"]:
                    index = int(detections[0, 0, i, 1])

                    # lọc các detections có label là person
                    if CLASSES[index] != "person":
                        continue

                    # tính toán tọa độ bounding box cho object
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    #tạo đối tượng tracker cho từng vật thể
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # thêm tracker vào danh sách trackers
                    trackers.append(tracker)

            #bỏ qua frame đầu, không cần vẽ bounding box

        #tracking các vật thể
        else:
            # lặp qua các tracker
            for tracker in trackers:
                # cập nhật tracker và lấy tọa độ bounding box mới
                tracker.update(rgb)
                pos = tracker.get_position()

                # lấy tọa độ bounding box
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # vẽ bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)







        totalFrames += 1
        fps.update()
        cv2.imshow("Frame", frame)
        print("Total frame: ", totalFrames)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    fps.stop()
    print("Thoi gian chay: {:.2f}".format(fps.elapsed()))
    print("FPS trung binh: {:.2f}".format(fps.fps()))

    # vd.release()
    cv2.destroyAllWindows()


main()