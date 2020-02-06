import cv2
from drowsiness_detection_image import get_eye_aspect
from yawn_detection_image import get_yawn
from pupil_position_image import get_pupil_pos




def get_attention_feature(frame, drowsy_threshold = 0.3, yawn_threshold = 30 ):
    attention_dict = {
            "drowsy":False,
            "yawn":False,
            "center":None}
    attention_dict["yawn"] = get_yawn(frame, yawn_threshold)
    attention_dict["drowsy"] = get_eye_aspect(frame, drowsy_threshold)
    attention_dict["center"] = get_pupil_pos(frame)
    
    return attention_dict
    
if __name__ == "__main__":
    frame =  cv2.imread("/home/parag/Pictures/Webcam/2020-02-06-095435.jpg")
    result = get_attention_feature(frame)
    print(result)
