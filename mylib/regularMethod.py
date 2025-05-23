import cv2

def getFPS(time, totalFrames):
    return "{:.2f}".format(totalFrames / time)


def putText(frame, text, coor, font = cv2.FONT_HERSHEY_SIMPLEX, size = 0.6, color = (0, 255, 0), thickness = 4):
    cv2.putText(frame, text, coor, font, size, color, thickness)