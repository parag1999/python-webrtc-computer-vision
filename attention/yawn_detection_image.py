import cv2
import numpy as np

def get_landmarks(im, detector, predictor):
    
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

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

def mouth_open(image, detector, predictor):
    landmarks = get_landmarks(image, detector, predictor)
    
    if landmarks == "error":
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

    #cv2.imshow('Result', image_with_landmarks)
    #cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


def get_yawn(frame, detector, predictor, threshold_lip_distance = 30):
    
    image_landmarks, lip_distance = mouth_open(frame, detector, predictor)
    
    
    if lip_distance > threshold_lip_distance:
        
        #cv2.putText(frame, "Subject is Yawning", (50,450), 
        #            cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        

        #output_text = " Yawn Count: " + str(yawns + 1)

        #cv2.putText(frame, output_text, (50,50),
        #            cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
        return True
        
    else:
        return False 
     
    #If you need to see an image
    #cv2.imshow('Live Landmarks', image_landmarks )
    #cv2.imshow('Yawn Detection', frame )
    #cv2.waitKey(10000)
    #cv2.destroyAllWindows()        
 

if __name__ == "__main__":
    frame =  cv2.imread("/home/parag/Pictures/Webcam/2020-02-05-095941.jpg")
    if(get_yawn(frame)):
        print("Yawning")
    else:
        print("Not Yawning")

