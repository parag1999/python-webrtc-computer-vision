import cv2
from drowsiness_detection_image import get_eye_aspect
from yawn_detection_image import get_yawn
from pupil_position_image import get_pupil_pos

def get_score(drowsy, yawn, center):
    score = 0
    if not drowsy:
        score +=1
    else:
        score +=0.25
    if not yawn:
        score+=1
    if center == None:
        score += 0.25
    elif center == True:
        score += 1
    return score


def get_attention_feature(frame, drowsy_threshold = 0.3, yawn_threshold = 30 ):
    drowsy=False
    yawn=False
    center=None
    score = 0
    yawn = get_yawn(frame, yawn_threshold)
    drowsy = get_eye_aspect(frame, drowsy_threshold)
    center = get_pupil_pos(frame)
    score = get_score(drowsy, yawn, center)
    
    return {"yawn":yawn, "drowsy":drowsy, "center":center, "score":score}
    
if __name__ == "__main__":
    frame =  cv2.imread("/home/parag/Pictures/Webcam/2020-02-06-094014.jpg")
    result = get_attention_feature(frame)
    print(result)
