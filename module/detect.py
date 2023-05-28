import numpy as np
import cv2, dlib
from mylib.trackableobject import TrackableObject

def DetectPeople(frame, net, args, trackers, W, H, rgb, LABELNAME):
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

    return trackers


def DetectDirection(objects, trackablObject, H):

    UP = 0
    DOWN = 0
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
                    UP+=1
                    to.counted = True

                elif direction > 0 and centroid[1] > H // 2:
                    DOWN+=1
                    to.counted = True

        trackablObject[objectID] = to

    return UP, DOWN