def TrackingObject(trackers, rgb):
    rects = []  # lưu tọa độ bounding box của các vật thể

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
        

    return rects