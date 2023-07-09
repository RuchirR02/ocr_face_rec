import cv2 #for wbcam

video = cv2.VideoCapture(0)
facesdetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
while True:
    ret,frame = video.read() #two values where ret has a boolean value whether our webcam is ok and 2nd is our frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facesdetect.detectMultiScale(gray,1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w, y+h), (50,50,255), 1)
    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

