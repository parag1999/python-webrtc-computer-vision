import cv2
import pathlib
import dlib
from .drowsiness_detection_image import get_eye_aspect
from .yawn_detection_image import get_yawn
from .pupil_position_image import get_pupil_pos

detector = dlib.get_frontal_face_detector()
folder = pathlib.Path(__file__).parent.absolute()
predictor = dlib.shape_predictor(f"{folder}/shape_predictor_68_face_landmarks.dat")

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


async def get_attention_feature(frame, drowsy_threshold = 0.3, yawn_threshold = 30, num=None ):
    drowsy=False
    yawn=False
    center=None
    score = 0
    yawn = get_yawn(frame, detector, predictor, yawn_threshold)
    drowsy = get_eye_aspect(frame, detector, predictor, drowsy_threshold)
    center = get_pupil_pos(frame, detector, predictor)
    score = get_score(drowsy, yawn, center)

    res = {"yawn":yawn, "drowsy":drowsy, "center":center, "score":score}
    print(f"{num} -> {res}")
    
    return res

# if __name__ == "__main__":
#     frame =  cv2.imread("/home/parag/Pictures/Webcam/2020-02-06-094014.jpg")
#     result = get_attention_feature(frame,drowsy_threshold = 0.3, yawn_threshold = 30, detector = detector, predictor = predictor)
#     print(result)
