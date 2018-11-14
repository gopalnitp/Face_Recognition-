
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
from sklearn.externals import joblib
path="/home/gopal/Desktop/face_reco/numpy_data/"
c_id=0
data_ap=[]
dd={}
lab=[]
cap=cv2.VideoCapture(1)
face_cascde=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
diR=os.listdir(path)
for ff in diR:
    dd[c_id]=ff[:-4]
    data_item=np.load(path+ff)
    data_ap.append(data_item)
    
    tar=c_id*np.ones((data_item.shape[0],))
    c_id+=1
    
    lab.append(tar)
    
face_dataset=np.concatenate(data_ap,axis=0)
face_lab=np.concatenate(lab,axis=0).reshape((-1,1))
print face_dataset.shape
print face_lab.shape
face_final=np.concatenate((face_dataset,face_lab),axis=1)
face=face_final[:,:10000]/255.0
from sklearn.ensemble import RandomForestClassifier
fr=RandomForestClassifier()
fr.fit(face,face_final[:,10000].astype('float'))
joblib.dump(fr, "gopal_model.sav")

