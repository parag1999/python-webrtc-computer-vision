import dlib
import cv2
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist

def calculate_eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    
    eye_aspect_ratio = (A+B)/(2.0*C)
    
    return eye_aspect_ratio

def get_eye_aspect(frame, detector, predictor, eye_aspect_ratio_threshold = 0.3):
    
    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    frame = imutils.resize(frame, width=450)
    frame = cv2.flip(frame, 1)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rectangles = detector(gray_image, 0) #0 denotes that dont subtract mean layers from the original image 

    for rect in rectangles:
        shape = predictor(gray_image, rect)
        shape = face_utils.shape_to_np(shape)
        
        #Slicing left and right eye from shape
        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]
        
        left_eye_aspect_ratio = calculate_eye_aspect_ratio(left_eye)
        right_eye_aspect_ratio = calculate_eye_aspect_ratio(right_eye)
        
        average_eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio)/2.0
        
        #If you need to see the image
        #cv2.putText(frame, "EYE ASPECT RATIO: {:.2f}".format(average_eye_aspect_ratio),(300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 2)
            
        #cv2.imshow("frame", frame)
        #cv2.waitKey(10000)
        #cv2.destroyAllWindows()     
        if average_eye_aspect_ratio < eye_aspect_ratio_threshold:
        
            return True
            
        else:
            
            return False
                        
    
if __name__ == "__main__":
    frame =  cv2.imread("/home/parag/Pictures/Webcam/2020-02-05-092935.jpg")
    if(get_eye_aspect(frame)):
        print("Drowsy")
    else:
        print("Not Drowsy")

