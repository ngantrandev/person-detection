import cv2
import dlib

import numpy as np
import time

import imutils
from imutils.video import FPS
from vidgear.gears import CamGear
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from mylib.regularMethod import getFPS, putText

#import các biến từ file config.py
from mylib.config import args
from mylib.config import LABELNAME

#duong link video
src = "https://youtu.be/d0mcaxedv_4"
# src = "https://youtu.be/Y4XJx6aRH0I"
stream_mode = True   #True nếu muốn lấy video từ internet, False nếu muốn lấy video từ local

def main():
    #khởi tạo đối tượng FPS để đo tốc độ xử lý
    fps = FPS().start()
    # lưu trữ kích thước khung hình, lấy từ frame đầu tiên
    H = 0
    W = 0
    totalFrames = 0
    curr_FPS = 0

    # read cafe model from mobilenet_ssd folder using args
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # list này lưu thông tin các đối tượng để tracking
    trackers = []
    trackablObject={}
    # tạo đối tượng dùng để tracking
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)

    
    vd = CamGear(source = src, stream_mode = stream_mode).start()

    t_start = time.time()

    # vd = cv2.VideoCapture("https://youtu.be/T2-3J2hWjbk")
    while True:
        frame = vd.read()
        # thay đổi kích thước khung hình thành 500px. càng ít dữ liệu thì xử lý càng nhanh
        frame = imutils.resize(frame, width = 500)
        # chuyển đổi khung hình sang RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #vì dlib cần RGB để xử lý

        # lấy kích thước khung hình từ frame đầu tiên
        if W == 0 or H == 0:
            (H, W) = frame.shape[:2]

        rects = [] # lưu tọa độ bounding box của các vật thể
        objects = {} # là từ điển chứa danh sách object được tracking

        # nhận diện vật thể
        if totalFrames % args["skip_frames"] == 0:
            trackers.clear()  # làm mới danh sách

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
                    if LABELNAME[index] != "person":
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
            
            t_current = time.time()
            curr_FPS = getFPS(t_current - t_start, totalFrames)
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

                rects.append((startX, startY, endX, endY))
                # vẽ bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            

            putText(frame, "FPS: "+curr_FPS, (10, 30))
            putText(frame, "Total Frames: "+str(totalFrames), (10, 60))


        objects = ct.update(rects)

        #lap qua cac doi tuong được tracking
        for(objectID, centroid) in objects.items():
            to = trackablObject.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)

            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[1] < H // 2:
                        to.counted = True

                    elif direction > 0 and centroid[1] > H // 2:
                        to.counted = True
            

            trackablObject[objectID] = to
            
            putText(frame, "ID {}".format(objectID), (centroid[0]-10, centroid[1]-10))
                        

            








        totalFrames += 1
        fps.update()

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break


    fps.stop()
    print("Thoi gian chay: {:.2f}".format(fps.elapsed()))
    print("FPS trung binh: {:.2f}".format(fps.fps()))

    # vd.release()
    cv2.destroyAllWindows()


main()