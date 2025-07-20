import cv2
TrainedDtatSet=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video=cv2.VideoCapture(0)
while True:
  success,frame=video.read()
  if success==True:
    gray_image= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=TrainedDtatSet.detectMultiScale(gray_image)
    for x,y,w,h in faces:
     cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
    cv2.imshow('video',frame)
    key=cv2.waitKey(1)
    if key == 81 or key == 133:
      break
  else:
    print("video complete or frame nil")
    break