import numpy as np
import pickle
import sklearn
import cv2

mean=pickle.load(open('./model/mean_preprocess.pickle','rb'))
model_svm=pickle.load(open('./model/model_svm.pickle','rb'))
model_pca=pickle.load(open('./model/pca_50.pickle','rb'))

#settings
gender_pred=['Male','Female']
font=cv2.FONT_HERSHEY_SIMPLEX
args={ 'prototxt': './model/deploy.prototxt.txt', 'model': './model/res10_300x300_ssd_iter_140000.caffemodel', 'confidence': 0.5}

def pipeline_model(img,filename,color='bgr'):
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    print('Model load successfully')
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    image = cv2.imread(img)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            

        if (startX, startY, endX, endY) ==():
            return False
     
        cv2.rectangle(image,(startX,startY),(endX,endY),(0,255,0),3)
        roi=image[startY:endY,startX:endX]
        roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        #step-4:normalization (0-1)
        roi=roi/255.0
        #step-5resize images(100,100)
        if roi.shape[1]>100:
            roi_resize=cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize=cv2.resize(roi,(100,100),cv2.INTER_CUBIC)
        #step-6: Flattening (1*10000)
       
        roi_reshape=roi_resize.reshape(1,-1)
        #step-7: subtract with mean
        mean1=mean.reshape(1,-1)
      
        roi_mean=roi_reshape-mean1
        #step-8: get eigen image
        eigen_image=model_pca.transform(roi_mean)
        #step-9: pass to ml model(svm)
        results=model_svm.predict_proba(eigen_image)[0]
        #step-10: 
        predict=results.argmax() #0 or 1
        score=results[predict]
        # #step-11:
        # text='%s: %0.2f' %(gender_pred[predict],score)
        text='%s: %0.2f' %(gender_pred[predict],score)
        cv2.putText(image,text,(startX,startY),font,1,(0,255,0),3)
    cv2.imwrite('static/predicts/{}'.format(filename),image)

