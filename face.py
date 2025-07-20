import cv2

#TrainedDataSet
TrainedDataSet=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Read a image
img=cv2.imread('deva.jpg/download (6).jpg')


#convert into grayscale
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=TrainedDataSet.detectMultiScale(gray)
print(faces)
for x,y,w,h in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
cv2.imshow('download',img)
cv2.imshow('gray',gray)
cv2.waitKey()  
