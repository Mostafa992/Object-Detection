import numpy as np
import cv2


############################################
classes_path="cocoNames/coco.names"
configuration_path="yolov3.cfg"
weights_path="yolov3.weights"
width,height=320,320
conf_threshold=0.5
nms_threshold=0.3
#########################################
with open(classes_path,"rt") as f:
    classes=f.read().rstrip("\n").split("\n")
##print("classes detected: \n",classes)
## print("Number of classes detected: ",len(classes))

# Read Network
net=cv2.dnn.readNetFromDarknet(configuration_path,weights_path)
#set Backend&Target
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


#Intiate VideoFeed
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)


def find_objects(output,img):
    H,W,C=img.shape
    bbox=[]
    class_Ids=[]
    confidences=[]

    for output_layer in output_layers:
        for detection in output_layer:
            scores=detection[5:]  #remove the first 5 elements
            class_Id=np.argmax(scores)
            confidence=scores[class_Id]

            if confidence>conf_threshold:
                w,h=int(detection[2]*W),int(detection[3]*H)
                x,y=int((detection[0]*W)-w/2),int((detection[1]*H)-h/2)

                bbox.append([x,y,w,h])
                class_Ids.append(class_Id)
                confidences.append(float(confidence))

    ##print("number of objects detected: ",len(bbox))

    #define the indices for the best fit bbox
    indices=cv2.dnn.NMSBoxes(bbox,confidences,conf_threshold,nms_threshold)
    for i in indices:
        i=i[0]
        box=bbox[i]
        x,y,w,h=box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classes[class_Ids[i]]}{int(confidences[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
while True:
    succ,frame=cap.read()

    #convert Frame to blob
    blob=cv2.dnn.blobFromImage(frame,1/255,(width,height),[0,0,0],1,crop=False)

    #Netword Input
    net.setInput(blob)
    #Network Output
    layer_names=net.getLayerNames()
    ##print(layer_names)
    ##print(len(layer_names))

    #Extract output layers only
    output_layers_idx=net.getUnconnectedOutLayers()
    ## print(output_layers_idx)
    ## print(len(output_layers_idx))
    output_layers_names=[layer_names[i[0]-1]for i in output_layers_idx]
    ##print(output_layers_names)


    #send image/frame as forward pass to the network and find output layers

    output_layers=net.forward(output_layers_names)
    ## print(output_layers)
    ##print(len(output_layers))
    ##print(output_layers[0].shape)
    ##print(output_layers[1].shape)
    ##print(output_layers[2].shape)
    #print(output_layers[0][0])

    find_objects(output_layers,frame)





    cv2.imshow("Frame",frame)

    if cv2.waitKey(1)&0xff==27:
        break