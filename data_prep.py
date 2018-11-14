
# coding: utf-8

# In[33]:


import cv2
import numpy as np


# In[36]:


import cv2
import numpy as np
inputname=input("your name")
path="/home/gopal/Desktop/face_reco/numpy_data/"
cap=cv2.VideoCapture(1)
face_cascde=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
skip=0
face_data=[]
while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    gray_fram=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces=face_cascde.detectMultiScale(frame,1.3,5)
       
    for face in faces:
        x,w,y,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        off=10
        intrested_face=gray_fram[y+off:y+h+off,x+off:x+w+off]
        intrested_face=cv2.resize(intrested_face,(100,100))
        skip+=1
        if skip%20==0:
            face_data.append(intrested_face)
            print len(face_data)
            
            
            
            
            
    #print faces    
    cv2.imshow("Frame",frame)
    
    key_press=cv2.waitKey(1) & 0xFF
    if key_press==ord('q'):
        break
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print face_data.shape
np.save(path+inputname+'.npy',face_data)
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

