
# coding: utf-8

# In[14]:


from sklearn.externals import joblib
import cv2
import numpy as np
import os


# In[15]:


fr = joblib.load("gopal_model.sav")


# In[16]:


path="/home/gopal/Desktop/face_reco/numpy_data/"


# In[19]:


dd={}
c_id=0
diR=os.listdir(path)
for ff in diR:
    dd[c_id]=ff[:-4]
    c_id=c_id+1


# In[20]:


dd


# In[21]:


cap=cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_SIMPLEX
face_cascde=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    gray_fram=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces=face_cascde.detectMultiScale(frame,1.3,5)
       
    for face in faces:
        x,w,y,h=face
        #print face
        off=10
        intrested_face1=gray_fram[y+off:y+h+off,x+off:x+w+off]
        intrested_face1=cv2.resize(intrested_face1,(100,100))
        gh=intrested_face1.flatten()
        gh=np.array(gh)/255.0
        gh=gh.reshape((-1,1)).T
        d=fr.predict(gh)
        #print fr.score(gh)
        name=dd[int(d[0])]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.putText(frame,name,(x,y+200), font, 1,(0,255,255),2,cv2.LINE_4)
        
        
        
    cv2.imshow('img',frame)
    key_press=cv2.waitKey(1) & 0xFF
    if key_press==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  

