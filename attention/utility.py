import numpy as np
from scipy.spatial import distance as dist

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

def calculate_eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    
    eye_aspect_ratio = (A+B)/(2.0*C)
    
    return eye_aspect_ratio

def get_landmarks(im, predictor, detector):
    


    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image, predictor, detector):
    landmarks = get_landmarks(image, predictor, detector)
    
    if landmarks == "error":
        return image, 0
    
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return lip_distance

