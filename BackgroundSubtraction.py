from locale import format_string
import cv2
import numpy as np
import time
import vehicles
import csv
import tensorflow as tf
import dlib
import vehicles
import centroidtracker 
from datetime import datetime
import os
from imutils.video import FPS
import imutils
import six

from centroidtracker import CentroidTracker
from vehicles import TrackableObject
from tensorflow_detection import DetectionObj

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils_modded as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

"""
Object Detection
"""
paths = {'CHECKPOINT_PATH':"Tensorflow/workspace/models/my_ssd_mobnet"}
files = {
    'PIPELINE_CONFIG': "Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config",
    'TF_RECORD_SCRIPT': "Tensorflow/scripts/generate_tfrecord.py", 
    'LABELMAP': "Tensorflow/workspace/annotations/label_map.pbtxt"
}
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-6')).expect_partial()

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

    

def vehicle_counting():
    frame_width = 640
    frame_height = 480
    videoSource = "C:\\Users\\anand\\Documents\\UTAR\\Y3S1\\FYP Project\\Code\\VideoSource\\demo_final.mp4"
    videoName= videoSource[64:-4]
    videoOut = f'C:\\Users\\anand\\Documents\\UTAR\\Y3S1\\FYP Project\\Code\\Result\\{videoName}_detection.mp4'
    LogOutPath = 'C:\\Users\\anand\\Documents\\UTAR\\Y3S1\\FYP Project\\Code\\Result'
    csv_name = f"{videoName}.csv"
    cap=cv2.VideoCapture(videoSource)     # Video Source
    out = cv2.VideoWriter(videoOut,cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame_width,frame_height)) #Write video to output file
    fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=False,history=200,varThreshold = 90)       # Create Foreground mask    
    kernalOp = np.ones((3,3),np.uint8)
    kernalOp2 = np.ones((5,5),np.uint8)
    kernalCl = np.ones((11,11),np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cars = []
    max_p_age = 5
    pid = 1
    cnt_left=0
    cnt_right=0
    cnt_left2=0
    cnt_right2=0
    cnt_car = 0
    cnt_motor = 0
    cnt_bicycle = 0
    cnt_nan = 0

    # maximum consecutive frames a given object is allowed to be marked as "disappeared" until we need to deregister the object from tracking
    max_disappear = 8
    # maximum distance between centroids to associate an object if the distance is larger than this maximum distance we'll start to mark the object as "disappeared"
    max_distance = 175
    #number of frames to perform object tracking instead of object detection
    track_object = 4
    #minimum confidence
    confidence = 0.4
    #frame width in pixels
    frame_width = 640
    #dictionary holding the different speed estimation columns
    speed_estimation_zone = (250,300, 350, 400)
    #real world distance in meters
    distance_left = 13 
    distance_right = 15.1
    #speed limit in kmph
    speed_limit = 50

    #Meter Per Pixel
    meterPerPixel_left = distance_left / frame_width
    meterPerPixel_right = distance_right/ frame_width


    # count the number of frames
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # start the frames per second throughput estimator
    fps_2 = int(cap.get(cv2.CAP_PROP_FPS))
    
    # calculate dusration of the video
    seconds = int(frames / fps_2)
    duration = float(seconds/3600)

    print(f"frames: {frames} fps: {fps_2} seconds:{seconds} duration:{duration}")
    print("Car counting and classification")

    line_left=310 #320 dv
    line_right=330 #450 dv

    left_limit=540 #470 for diagonal view (dv)
    right_limit=60 #300 for diagonal view

    # ---------------------------------- Speed Measurement Parameters ---------------------------------------
    ct = CentroidTracker(maxDisappeared= max_disappear)
    trackers = []
    trackableObjects = {}
    # keep the count of total number of frames
    frame_count = 0
    # initialize the log file
    logFile = None
    # initialize the list of various points used to calculate the avg of
    # the vehicle speed
    points = [("A", "B"), ("B", "C"), ("C", "D")]
    
    

    while(cap.isOpened()):
        timez = float(frame_count/fps_2)
        ret,frame=cap.read()
        ts = datetime.now()
        newDate = ts.strftime("%m-%d-%y")
        rects = []
        centroidz = []
        
        
        

        if frame is None:
            break

        if logFile is None:
            # build the log file path and create/open the log file
            logPath = os.path.join(LogOutPath, csv_name)
            logFile = open(logPath, mode="a")
            # set the file pointer to end of the file
            pos = logFile.seek(0, os.SEEK_END)
            # if we are using dropbox and this is a empty log file then
            # write the column headings
           
            if pos == 0:
                logFile.write("Year,Month,Day,Time, ObjectID ,Direction, Vehicle Class, Speed(km/h)\n")
        

       

        # Object Detection
        image_np = np.array(frame)
    
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, axis=0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        class_label, rects2, _ = viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.8,
                    agnostic_mode=False)

        image_np_with_detections = cv2.resize(image_np_with_detections, (frame_width, frame_height))
            

        for i in cars:
            i.age_one()
        fgmask=fgbg.apply(frame)
        if frame_count == 1080:
            cv2.imwrite(f'C:\\Users\\anand\\Documents\\UTAR\\Y3S1\\FYP Project\\Code\\Result\\maskSub_{frame_count}.png',fgmask)

        

        

        if ret==True: 
            ret,imBin=cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
            # cv2.imwrite(f'C:\\Users\\anand\\Documents\\UTAR\\Y3S1\\FYP Project\\Code\\Result\\image_thresh_{frame_count}.png',imBin)
            mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernalOp)
            # cv2.imwrite(f'C:\\Users\\anand\\Documents\\UTAR\\Y3S1\\FYP Project\\Code\\Result\\image_morphOp_{frame_count}.png',mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernalCl)
            # cv2.imwrite(f'C:\\Users\\anand\\Documents\\UTAR\\Y3S1\\FYP Project\\Code\\Result\\image_morphClose_{frame_count}.png',mask)


            (countours0,hierarchy)=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for cnt in countours0:
                area=cv2.contourArea(cnt)
              

                if (area>2000 and area < 60000):

                    m=cv2.moments(cnt)
                    cx=int(m['m10']/m['m00'])
                    cy=int(m['m01']/m['m00'])
                    x,y,w,h=cv2.boundingRect(cnt)

                    rects.append((x, y, w, h))
                    centroidz.append((cx,cy))
                    # add the bounding box coordinates to the rectangles list
                    

                    # use the centroid tracker to associate the (1) old object
                    # centroids with (2) the newly computed object centroids

                    new=True
                    if cx in range(right_limit,left_limit): # If car within limit
                        for i in cars:
                            if abs(x - i.getX()) <= w and  abs(y - i.getY()) <= h :  #Check whether new car or old car that has moved
                                new = False
                                i.updateCoords(cx, cy) #Update Coordinate if car moves

                                # Determine the direction of the car
                                if i.going_LEFT(line_right,line_left)==True:
                                    cnt_left+=1

                                elif i.going_RIGHT(line_right,line_left)==True:
                                    cnt_right+=1

                                   

                                break

                            # If reach limit stop bounding rectangle
                            if i.getState()=='1':
                                if i.getDir()=='right'and i.getX()>right_limit:
                                    i.setDone()
                                elif i.getDir()=='left'and i.getX()<left_limit:
                                    i.setDone()
                            if i.timedOut():
                                index=cars.index(i)
                                cars.pop(index)
                                del i

                        if new==True:
                            p=vehicles.Car(pid,cx,cy,max_p_age)
                            cars.append(p)
                            pid+1
                    cv2.circle(image_np_with_detections, (cx, cy), 2, (0, 0, 255), -1) # Draw the centroid of the car


                    img=cv2.rectangle(mask,(x,y),(x+w,y+h),(244,255,100),2) # Draw rectangle over the car
                   
                    if frame_count == 1080:
                        frame=cv2.line(frame,(line_left,0),(line_left,480),(0,0,255),3,8)
                        frame=cv2.line(frame,(left_limit,0),(left_limit,480),(255,255,0),1,8) # Display left limit

                        frame=cv2.line(frame,(right_limit,0),(right_limit,480),(255,255,0),1,8) # Display right limit
                        frame = cv2.line(frame, (line_right, 0), (line_right, 480), (255, 0,0), 3, 8)
                        cv2.imwrite(f'C:\\Users\\anand\\Documents\\UTAR\\Y3S1\\FYP Project\\Code\\Result\\imageFindcontour_{frame_count}.png',frame)




            objects = ct.update(rects, centroidz)
            
                    
            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                # check to see if a trackable object exists for the current
                # object ID
                to = trackableObjects.get(objectID, None)
                # print(f"Object ID:{objectID} centroid:{centroid}" )
               
                # if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(objectID, centroid)
                    to.vehicleClass = class_label
                

                   
                    

                # otherwise, if there is a trackable object and its speed has
                # not yet been estimated then estimate it
                elif not to.estimated:
                    if to.vehicleClass is None or to.vehicleClass == "None":
                        to.vehicleClass = class_label
                    # check if the direction of the object has been set, if
                    # not, calculate it, and set it
                    if to.direction is None or to.direction == 0:
                        y = [c[0] for c in to.centroids]
                        direction = centroid[0] - np.mean(y)
                        to.direction = direction

                        # print(f"npmeany: {np.mean(y)} direction: {direction}")

                            # if the direction is positive (indicating the object
                    # is moving from left to right)
                    if to.direction > 0:
                        # check to see if timestamp has been noted for
                        # point A
                        if to.timestamp["A"] == 0 :
                            # if the centroid's x-coordinate is greater than
                            # the corresponding point then set the timestamp
                            # as current timestamp and set the position as the
                            # centroid's x-coordinate
                            if centroid[0] > speed_estimation_zone[0]:
                                to.timestamp["A"] = timez
                                to.position["A"] = centroid[0]
                        # check to see if timestamp has been noted for
                        # point B
                        elif to.timestamp["B"] == 0:
                            # if the centroid's x-coordinate is greater than
                            # the corresponding point then set the timestamp
                            # as current timestamp and set the position as the
                            # centroid's x-coordinate
                            if centroid[0] > speed_estimation_zone[1]:
                                to.timestamp["B"] = timez
                                to.position["B"] = centroid[0]
                        # check to see if timestamp has been noted for
                        # point C
                        elif to.timestamp["C"] == 0:
                            # if the centroid's x-coordinate is greater than
                            # the corresponding point then set the timestamp
                            # as current timestamp and set the position as the
                            # centroid's x-coordinate
                            if centroid[0] > speed_estimation_zone[2]:
                                to.timestamp["C"] = timez
                                to.position["C"] = centroid[0]
                        # check to see if timestamp has been noted for
                        # point D
                        elif to.timestamp["D"] == 0:
                            # if the centroid's x-coordinate is greater than
                            # the corresponding point then set the timestamp
                            # as current timestamp, set the position as the
                            # centroid's x-coordinate, and set the last point
                            # flag as True
                            if centroid[0] > speed_estimation_zone[3]:
                                to.timestamp["D"] = timez
                                to.position["D"] = centroid[0]
                                to.lastPoint = True

                    # if the direction is negative (indicating the object
                    # is moving from right to left)
                    elif to.direction < 0:
                        # check to see if timestamp has been noted for
                        # point D
                        if to.timestamp["D"] == 0 :
                            # if the centroid's x-coordinate is lesser than
                            # the corresponding point then set the timestamp
                            # as current timestamp and set the position as the
                            # centroid's x-coordinate
                            if centroid[0] < speed_estimation_zone[0]:
                                to.timestamp["D"] = timez
                                to.position["D"] = centroid[0]
                        # check to see if timestamp has been noted for
                        # point C
                        elif to.timestamp["C"] == 0:
                            # if the centroid's x-coordinate is lesser than
                            # the corresponding point then set the timestamp
                            # as current timestamp and set the position as the
                            # centroid's x-coordinate
                            if centroid[0] < speed_estimation_zone[1]:
                                to.timestamp["C"] = timez
                                to.position["C"] = centroid[0]
                        # check to see if timestamp has been noted for
                        # point B
                        elif to.timestamp["B"] == 0:
                            # if the centroid's x-coordinate is lesser than
                            # the corresponding point then set the timestamp
                            # as current timestamp and set the position as the
                            # centroid's x-coordinate
                            if centroid[0] < speed_estimation_zone[2]:
                                to.timestamp["B"] = timez
                                to.position["B"] = centroid[0]
                        # check to see if timestamp has been noted for
                        # point A
                        elif to.timestamp["A"] == 0:
                            # if the centroid's x-coordinate is lesser than
                            # the corresponding point then set the timestamp
                            # as current timestamp, set the position as the
                            # centroid's x-coordinate, and set the last point
                            # flag as True
                            if centroid[0] < speed_estimation_zone[3]:
                                to.timestamp["A"] = timez
                                to.position["A"] = centroid[0]
                                to.lastPoint = True

                    # check to see if the vehicle is past the last point and
                    # the vehicle's speed has not yet been estimated, if yes,
                    # then calculate the vehicle speed and log it if it's
                    # over the limit
                    if to.lastPoint and not to.estimated:
                        # print(to.position["A"], to.position["B"], to.position["C"], to.position["D"])
                        # initialize the list of estimated speeds
                        estimatedSpeeds = []

                        if to.vehicleClass is None or to.vehicleClass == "None":
                            # if class_label is None:
                            #     to.vehicleClass = "motorcycle"
                            #     cnt_motor += 1
                            # else:
                            to.vehicleClass = class_label

                        if to.vehicleClass == "car":
                            cnt_car+=1
                        elif to.vehicleClass =="motorcycle":
                            cnt_motor+=1
                        elif to.vehicleClass =="bicycle":
                            cnt_bicycle+=1
                        else:
                            cnt_nan+=1
                        
                        # loop over all the pairs of points and estimate the
                        # vehicle speed
                        for (i, j) in points:
                            # calculate the distance in pixels
                            d = to.position[j] - to.position[i]
                            distanceInPixels = abs(d)
                            # print(f"Distance In Pixel: {distanceInPixels}")
                            # check if the distance in pixels is zero, if so,
                            # skip this iteration
                            if distanceInPixels == 0:
                                continue
                            # calculate the time in hours
                            if to.timestamp[j] is str or to.timestamp[i] is str:
                                estimatedSpeeds.append(100)
                            else:
                                # print(f"Timestamp J: {type(to.timestamp[j])} Timestamp I: {type(to.timestamp[i])}")
                                timeInSeconds = abs(to.timestamp[j] - to.timestamp[i])
                            
                                # timeInSeconds = abs(t.total_seconds())
                                timeInHours = timeInSeconds / (60 * 60)
                                # calculate distance in kilometers and append the
                                # calculated speed to the list
                                if direction > 0:
                                    distanceInMeters = distanceInPixels * meterPerPixel_right
                                elif direction < 0:
                                    distanceInMeters = distanceInPixels * meterPerPixel_left

                                distanceInKM = distanceInMeters / 1000
                                estimatedSpeeds.append(distanceInKM / timeInHours)
                                
                            # else: 
                            #     estimatedSpeeds.append(100)
                        # calculate the average speed
                        if to.direction < 0:
                            cardirectionz = "Left"
                            
                        elif to.direction > 0:
                            cardirectionz = "Right"
                            

                        to.calculate_speed(estimatedSpeeds)
                        # set the object as estimated
                        to.estimated = True
                        print("[INFO] Speed of the vehicle that just passed"\
                            " is: {:.2f} KMPH ObjectID: {} Direction: {} Class: {}".format(to.speedKMPH, objectID, cardirectionz, to.vehicleClass))
                        # textz = "Speed: {:.2f}".format(to.speedKMPH)
                        # cv2.putText(image_np_with_detections, textz, (centroid[0] - 15, centroid[1] - 10)
                        # , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                # store the trackable object in our dictionary
                trackableObjects[objectID] = to

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(image_np_with_detections, text, (centroid[0] - 10, centroid[1] - 10)
                    , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(image_np_with_detections, (centroid[0], centroid[1]), 4,
                    (0, 255, 0), -1)

                # check if the object has not been logged
                if not to.logged:
                    # check if the object's speed has been estimated and it
                    # is higher than the speed limit
                    if to.estimated:
                        # set the current year, month, day, and time
                        year = ts.strftime("%Y")
                        month = ts.strftime("%m")
                        day = ts.strftime("%d")
                        time = ts.strftime("%H:%M:%S")
                        if to.direction < 0:
                            cardirection = "Left"
                            cnt_left2 +=1
                        elif to.direction > 0:
                            cardirection = "Right"
                            cnt_right2+=1
                        
                        # if to.vehicleClass == "car":
                        #     cnt_car+=1
                        # elif to.vehicleClass =="motorcycle":
                        #     cnt_motor+=1
                        # elif to.vehicleClass =="bicycle":
                        #     cnt_bicycle+=1
                        

                    
                        # log the event in the log file
                        info = "{},{},{},{},{},{},{},{:.2f}\n".format(year, month, day, time, to.objectID, cardirection,
                            to.vehicleClass, to.speedKMPH)
                        
                        logFile.write(info)
                        # set the object has logged
                        to.logged = True

                        



            #--------------------------Display info on video -------------------------------------------------------------

            # str_left='Going Right: '+str(cnt_left)
            # str_right='Going Left: '+str(cnt_right)
            # # frame=cv2.line(frame,(line_left,0),(line_left,480),(0,0,255),3,8)
            # # frame=cv2.line(frame,(left_limit,0),(left_limit,480),(255,255,0),1,8) # Display left limit

            # # frame=cv2.line(frame,(right_limit,0),(right_limit,480),(255,255,0),1,8) # Display right limit
            # # frame = cv2.line(frame, (line_right, 0), (line_right, 480), (255, 0,0), 3, 8)

            # # cv2.putText(frame, str_left, (110, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.putText(frame, str_right, (110, 40), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            # out.write(frame) # Export frame to video
            # cv2.imshow(videoName,frame) #Show vehicle counting
            # # cv2.imshow('FGmask', fgmask) #Show foreground mask

            str_left='Going Right: '+str(cnt_left)
            str_right='Going Left: '+str(cnt_right)
            # str_left2='Going Left TFOD: '+str(cnt_left2)
            # str_right2='Going right TFOD: '+str(cnt_right2)
            

            cv2.putText(image_np_with_detections, str_left, (110, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image_np_with_detections, str_right, (110, 60), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(image_np_with_detections, str_left2, (400, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.putText(image_np_with_detections, str_right2, (400, 60), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            out.write(image_np_with_detections) # Export frame to video
            # cv2.imshow(videoName,frame) #Show vehicle counting
            cv2.imshow('object detection',  image_np_with_detections)
            # cv2.imshow('FGmask', fgmask) #Show foreground mask    
        
            
            #-----------------When to End Video-----------------------
            frame_count += 1

            if cv2.waitKey(10)&0xff==ord('q'):
                break

        else:
            break

        
    
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if logFile is not None:
        infoz = "Total Vehicle: {},Total Left: {},Total Right: {},Total Cars: {}, Total Motorcycles: {}, Total Bicycles: {}\n".format((cnt_left + cnt_right), cnt_right, cnt_left, cnt_car, cnt_motor, cnt_bicycle)           
        logFile.write(infoz)

        logFile.close()




def detect_video():
    detection = DetectionObj(model='my_ssd_mobnet')
    detection.video_pipeline(video="C:\\Users\\anand\\Documents\\UTAR\\Y3S1\\FYP Project\\Code\\VideoSource\\video5.2.mp4", audio=False)


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


if __name__ == '__main__':
    # AI Classification counting Parameters to set 
    # modelPath = ""
    # lmapPath = ""
    # videoPath = "C:\\Users\\anand\\Documents\\UTAR\\Y3S1\\FYP Project\\Code\\VideoSource\\video5.2.mp4"
    # threshold = 111

    # #tensorflow Parameters
    # WORKSPACE_PATH = 'C:/Users/anand/Documents/GitHub/RealTimeObjectDetection/Tensorflow/workspace'
    # ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
    # IMAGE_PATH = WORKSPACE_PATH+'/images'
    # MODEL_PATH = WORKSPACE_PATH+'/models'
    # PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
    # CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
    # CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/' 

    vehicle_counting()
    
  

   