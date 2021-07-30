
# import standard libraries
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import dlib
from imutils.video import FPS
import pandas as pd

# import libraries for tracking
from tracking.centroidtracker import CentroidTracker
from tracking.trackableobject import TrackableObject

# Initialize the parameters
confThreshold = 0.3  #Confidence threshold
nmsThreshold = 0.2   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image
skip_frames = 15     #No. of frames skipped for next detection

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser(description='Object Detection and Tracking using YOLO in OPENCV')
parser.add_argument("-v",'--video', help='Path to video file.')
parser.add_argument("-s", "--skip_frames", type=int, default=5,
    help="# of skip frames between detections")
args = parser.parse_args()
        
# Load names of classes
classesFile = "model/coco.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "model/yolov3-tiny_obj.cfg";
modelWeights = "model/yolov3-tiny1.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

totalEntering = 0
totalLeaving = 0



output = pd.read_csv('output.csv')


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def MarkPeople(objects, total, totalEntering, totalLeaving):

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        else:

            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)

            to.centroids.append(centroid)


            # check to see if the object has been counted or not
            if not to.counted:

                if direction < 0 and centroid[1] < H // 2:

                    
                    totalLeaving += 1
                    to.counted = True



                elif direction > 0 and centroid[1] > H // 2:
                    totalEntering += 1
                    to.counted = True

                # total+=1
                # to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "Person {}".format(objectID)
        cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        

    return total, totalEntering, totalLeaving
    

# Remove the bounding boxes with low confidence using non-maxima suppression
def Fill_tracker_list(frame, outs, count):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    trackers = []

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            assert(classId < len(classes))
            confidence = scores[classId]

            # Check if confidence is more than threshold and the detected object is a person
            if(confidence > confThreshold and classes and classes[classId]=="person"):
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(left, top, left+width, top+height)
        tracker.start_track(rgb, rect)
        # add the tracker to our list of trackers so we can
        # utilize it during skip frames
        trackers.append(tracker)
        #drawPred(classIds[i], confidences[i], left, top, left + width, top + height, count)
    return trackers

# Process inputs
winName = 'People Counter'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"

if (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = 'output/' + args.video.split('/')[1][:-4]+'_output.avi'

else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
writer = None


# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=20, maxDistance=40)
trackers = []
trackableObjects = {}
total = 0

# start the frames per second throughput estimator
fps = FPS().start()
totalFrames = 0

while True:
    
    # get frame from the video
    hasFrame, frame = cap.read()
    
    #converting frame form BGR to RGB for dlib 
    # rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    rgb = frame

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break

    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % args.skip_frames == 0:
        
        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence and store in trackers list for future tracking
        trackers = Fill_tracker_list(frame, outs, 0)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        # t, _ = net.getPerfProfile()
        # label = 'Inference time: %.2f ms/ Count: %d' % (t * 1000.0 / cv.getTickFrequency(), count)
        # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))


    objects = ct.update(rects)

    # Mark  all the persons in the frame


    # totalEnter, totalLeft, total = MarkPeople(objects, total)

    total, totalEntering, totalLeaving = MarkPeople(objects, total, totalEntering, totalLeaving)
    
    # construct a tuple of information we will be displaying on the
    # frame
    info = [
        ("In ", totalEntering),
        ("Out ", totalLeaving),
    ]

    # loop over the info tuples and draw them on our frame
    H = None
    W = None

    (H, W) = frame.shape[:2]
    for (i, (k, v)) in enumerate(info):
        text = "{}:: {}".format(k, v)
        cv.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)


    # show the output frame
    # Write the frame with the detection boxes
    if (writer is None):
        fourcc = cv.VideoWriter_fourcc(*"MJPG")

        writer = cv.VideoWriter(outputFile, fourcc, 30,
        (frame.shape[1], frame.shape[0]), True)


    writer.write(frame)

    cv.imshow(winName, cv.resize(frame, (1200,900)))
    key = cv.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1
    fps.update()


new_data = {'Video recording name': args.video.split('/')[1], 'Category':'Pedestrians coming in', 'Count': totalEntering}
new_data2 = {'Video recording name': args.video.split('/')[1], 'Category':'Pedestrians going out', 'Count': totalLeaving}

output = output.append(new_data, ignore_index=True)
output = output.append(new_data2, ignore_index=True)


output.to_csv('output.csv')

# stop the timer and display FPS information
fps.stop()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv.destroyAllWindows()
