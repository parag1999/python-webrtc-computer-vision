import cv2
from .gaze_tracking import GazeTracking

def get_pupil_pos(frame, detector, predictor):
    gaze = GazeTracking(detector=detector, predictor = predictor)
    
    frame = cv2.flip(frame,1)
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    
    if not left_pupil or not right_pupil:
        return None
    if gaze.is_center():
        return True
    else:
        return False
    #cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    #cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    #cv2.imshow("Demo", frame)
    #cv2.waitKey(10000)
    #cv2.destroyAllWindows() 


if __name__ == "__main__":
    frame = cv2.imread("/home/parag/Pictures/Webcam/2020-02-06-095035.jpg")
    print(get_pupil_pos(frame))
