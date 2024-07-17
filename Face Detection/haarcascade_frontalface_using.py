import cv2

cap = cv2.VideoCapture(0)

while True:    
    ret, frame = cap.read()    
    
    # Gri tonlamaya çevirme (göz tespiti için genellikle gri tonlamada çalışılır)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_model = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    
    faces = face_model.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)
    
    download_path_eye = "./haarcascade_eye.xml"
    
    eye_model = cv2.CascadeClassifier(download_path_eye)
    
    eyes = eye_model.detectMultiScale(gray_frame, scaleFactor=1.4, minNeighbors=4)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
    
    
    for (x,y,w,h) in eyes:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),1)
    
    cv2.imshow("img", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
        
cap.release()
cv2.destroyAllWindows()
