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

src = "videos/vidu2.mp4"
src = "https://youtu.be/FanKlgWthvk"
src = "https://www.youtube.com/live/_5pLLRT5FuM?feature=share"

stream_mode = True   #True nếu muốn lấy video từ internet, False nếu muốn lấy video từ local

LABELNAME = LABELS = open(args["label"]).read().strip().split("\n")

def main():
    # height và width của khung hình
    H = 0
    W = 0
    totalFrames = 0
    totalToLeft = 0
    totalToRight = 0
    currPeople = 0
    curr_FPS = 0

    rects = [] # lưu tọa độ bounding box của các vật thể
    objects = {} # là từ điển chứa danh sách object được tracking

    # read cafe model from mobilenet_ssd folder using args
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    # tạo đối tượng dùng để tracking
    ct = CentroidTracker(maxDisappeared=10, maxDistance=10)

    # list này lưu thông tin các đối tượng để tracking
    trackers = []
    trackablObject={}

    

    if args["stream"] == "true" or args["stream"] == "True":
        vd = CamGear(source = src, stream_mode = stream_mode).start()
        
    else:
        vd=CamGear().start()

    

    #khởi tạo đối tượng FPS để đo tốc độ xử lý
    fps = FPS().start()
    t_start = time.time()

    while True:
        currPeople = 0

        frame = vd.read()
        frame = imutils.resize(frame, width = 500)# thay đổi kích thước khung hình thành 500px. càng ít dữ liệu thì xử lý càng nhanh
        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #vì dlib cần RGB để xử lý

        if W == 0 or H == 0:
            (H, W) = frame.shape[:2]


        status = "Waiting"

        # nhận diện vật thể
        if totalFrames % args["skip_frames"] == 0:
            cv2.circle(frame, (W - 10, 10), 7, (0, 0, 255), thickness=7)
            status = "Detecting"
            # nhận diện vật thể
            trackers = DetectPeople(frame, net, args, W, H, rgb, LABELNAME)

        #tracking các vật thể
        else:
            if len(trackers) != 0:
                status = "Tracking"
                rects = TrackingObject(trackers, rgb)

                for rect in rects:
                    startX, startY, endX, endY = rect
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.circle(frame, (int((startX + endX)/2), int((startY+endY)/2)), 4, (0, 255, 0), -1)

        cv2.line(frame, (W//2, 0), (W//2, H), (0, 0, 0), 3)

        # xác định hướng di chuyển của các vật thể
        objects = ct.update(rects)
        toLeft, toRight, trackablObject = DetectDirection(objects, trackablObject, W)


        totalToLeft += toLeft
        totalToRight += toRight
        t_current = time.time()
        totalFrames += 1
        curr_FPS = getFPS(t_current - t_start, totalFrames)
        currPeople = len(trackers)

        putText(frame, "FPS: " + curr_FPS, (10, 30))
        putText(frame, "Total Frames: "+str(totalFrames), (10, 60))
        putText(frame, "Total People: "+str(currPeople), (10, H-10))
        putText(frame, "To Left {}".format(totalToLeft), (10,90))
        putText(frame, "To Right {}".format(totalToRight), (10,120))
        putText(frame, "Status: {}".format(status), (10,H-40))
        
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