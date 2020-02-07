
import dlib
import cv2
from .utility import calculate_eye_aspect_ratio, mouth_open, get_score
from .gaze_tracking import GazeTracking
import os




def get_analysis(file_location, drowsy_threshold = 0.3, yawn_threshold = 30): 
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.getcwd()+"/attention_feature/shape_predictor_68_face_landmarks.dat")
    gaze = GazeTracking(detector=detector, predictor = predictor)
    
    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    final_list = []
    webcam = cv2.VideoCapture(file_location)
    
    while (webcam.isOpened()):
        ret, frame = webcam.read()
        x = webcam.get(cv2.CAP_PROP_POS_MSEC)
        curr_frame = int(webcam.get(cv2.CAP_PROP_POS_FRAMES))
        if(ret):
            if curr_frame % 5 == 0:
                #frame = cv2.flip(frame,1)
                drowsy, yawn, center = False, False, None
                lip_distance = mouth_open(frame, predictor, detector)
            
                #frame = imutils.resize(frame, width=450)
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                gaze.refresh(frame)
            
                frame = gaze.annotated_frame()
                
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
                    
                    #Display
                    left_eye_hull = cv2.convexHull(left_eye) # convexHull returns a boundary around our object
                    right_eye_hull = cv2.convexHull(right_eye)
                    
                    cv2.drawContours(frame, [left_eye_hull], -1, (0,255,0), 2)
                    cv2.drawContours(frame, [right_eye_hull], -1, (0,255,0), 2)
                    
                    if average_eye_aspect_ratio < drowsy_threshold:
                        drowsy = True
                        #cv2.putText(frame, "DROWSINESS ALERT", (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 0, 255), 2)
                                 
                    #cv2.putText(frame, "EYE ASPECT RATIO: {:.2f}".format(average_eye_aspect_ratio),(300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1)
                
                if not isinstance(lip_distance, tuple):     
                        if lip_distance > yawn_threshold:
                            yawn = True
                            #cv2.putText(frame, "Subject is Yawning", (50,450), 
                            #            cv2.FONT_HERSHEY_COMPLEX, 0.8,(0,0,255),2)
                #text = ""
                left_pupil = gaze.pupil_left_coords()
                right_pupil = gaze.pupil_right_coords()
                
                if not left_pupil or not right_pupil:
                    center = None
    
                if gaze.is_center():
                    center = True
                else:
                    center = False
                score = get_score(drowsy, yawn, center)
                y = {"drowsy":drowsy, "yawn":yawn, "center":center, "score":score}
                #cv2.putText(frame, text, (200, 300), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
            
                #cv2.putText(frame, "Left pupil:  " + str(left_pupil), (200, 370), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)
                #cv2.putText(frame, "Right pupil: " + str(right_pupil), (200, 405), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)
            
                    
                #cv2.imshow("frame", frame)
                #key = cv2.waitKey(1) & 0xFF
                #if key == ord("q"):
                #    break        #Closes the frame
            #else:
            #    pass                
                #webcam.release()
                #cv2.destroyAllWindows()
                final_list.append([x/1000,y])
        else:
        
            webcam.release()
    return final_list
    #cv2.destroyAllWindows()  
    
