import cv2
import time
import imutils
from imutils.video import FPS
from vidgear.gears import CamGear

from mylib.centroidtracker import CentroidTracker
from mylib.regularMethod import getFPS, putText
from module.detect import DetectPeople, DetectDirection
from module.tracking_object import TrackingObject

#import các biến từ file config.py
from mylib.config import args

#duong link video
src = "https://youtu.be/d0mcaxedv_4"
# src = "https://youtu.be/Y4XJx6aRH0I"
# src = "https://youtu.be/EBb3XWr-LK4"
# src = "https://www.youtube.com/live/_5pLLRT5FuM?feature=share"
src = "videos/vidu2.mp4"
stream_mode = False   #True nếu muốn lấy video từ internet, False nếu muốn lấy video từ local

LABELNAME = LABELS = open(args["label"]).read().strip().split("\n")

def main():
    
    # lưu trữ kích thước khung hình, lấy từ frame đầu tiên
    H = 0
    W = 0
    totalFrames = 0
    totalUP = 0
    totalDOWN = 0
    currPeople = 0
    curr_FPS = 0

    rects = [] # lưu tọa độ bounding box của các vật thể
    objects = {} # là từ điển chứa danh sách object được tracking

    # read cafe model from mobilenet_ssd folder using args
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    # tạo đối tượng dùng để tracking
    ct = CentroidTracker(maxDisappeared=10, maxDistance=50)

    # list này lưu thông tin các đối tượng để tracking
    trackers = []
    trackablObject={}

    
    vd = CamGear(source = src, stream_mode = stream_mode).start()
    #khởi tạo đối tượng FPS để đo tốc độ xử lý
    fps = FPS().start()
    t_start = time.time()

    # vd = cv2.VideoCapture("https://youtu.be/T2-3J2hWjbk")
    while True:
        currPeople = 0

        frame = vd.read()
        frame = imutils.resize(frame, width = 500)# thay đổi kích thước khung hình thành 500px. càng ít dữ liệu thì xử lý càng nhanh
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #vì dlib cần RGB để xử lý

        if W == 0 or H == 0:
            (H, W) = frame.shape[:2]

        # nhận diện vật thể
        if totalFrames % args["skip_frames"] == 0:
            trackers.clear()  # làm mới danh sách

            # nhận diện vật thể
            trackers = DetectPeople(frame, net, args, trackers, W, H, rgb, LABELNAME)

        #tracking các vật thể
        else:
            rects = TrackingObject(trackers, rgb, frame)

            for rect in rects:
                startX, startY, endX, endY = rect
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            


        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)

        # xác định hướng di chuyển của các vật thể
        objects = ct.update(rects)
        UP, DOWN = DetectDirection(objects, trackablObject, H)

        totalUP += UP
        totalDOWN += DOWN
            

        t_current = time.time()
        curr_FPS = getFPS(t_current - t_start, totalFrames)

        currPeople = trackers.__len__()

        totalFrames += 1
        putText(frame, "FPS: " + curr_FPS, (10, 30))
        putText(frame, "Total Frames: "+str(totalFrames), (10, 60))
        putText(frame, "Total People: "+str(currPeople), (10, H-10))
        putText(frame, "DOWN {}".format(totalDOWN), (10,90))
        putText(frame, "UP {}".format(totalUP), (10,120))

        
        fps.update()

        cv2.imshow("Frame", frame)

        if cv2.waitKey(50) & 0xFF == 27:
            break


    fps.stop()
    print("Thoi gian chay: {:.2f}".format(fps.elapsed()))
    print("FPS trung binh: {:.2f}".format(fps.fps()))

    # vd.release()
    cv2.destroyAllWindows()


main()