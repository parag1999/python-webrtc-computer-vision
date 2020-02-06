import cv2
from drowsiness_detection_image import get_eye_aspect
from yawn_detection_image import get_yawn




def get_attention_feature(frame, drowsy_threshold = 0.3, yawn_threshold = 30 ):
    attention_dict = {
            "drowsy":False,
            "yawn":False}
    if(get_yawn(frame, yawn_threshold)):
        attention_dict["yawn"] = True
    if(get_eye_aspect(frame, drowsy_threshold)):
        attention_dict["drowsy"] = True
    
    return attention_dict
    
if __name__ == "__main__":
    frame =  cv2.imread("/home/parag/Pictures/Webcam/2020-02-05-092935.jpg")
    result = get_attention_feature(frame)
    print(result)
