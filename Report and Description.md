# People Counter Solution by Team Algorithm

This is a Computer Vision Application that counts the number of people entering and exiting a large commercial establishment.
We have developed just the Software Solution for this particular Challenge.

![Demo](/demo/Demo.gif)

# Tiny YOLOv3 was the model that was used for this challenge

## How did we solve this?
As required, we had to train the model on the videos that were provided.
To do this we had to create images from the videos that were given and train them on the model.Sample images were also obtained [here](https://drive.google.com/file/d/1H-GImz4_wFNbrgoKO53-6aQdBGNz5QFW/view?usp=drivesdk) .The total dataset used for training and testing can be found [here](https://drive.google.com/file/d/1LpdHUv5fx4Lbaa_CqMGGeokAZ29neekH/view?usp=sharing) to further train the model.
We realised that when images with higher quality are trained on the model,it is not able to effectively count the number of people entering and leaving the premises.We then focused on having training sets mainly based on the images that were generated from the videos that were provided.In all, we had about 450 images which was used to train the model. We believe with more data and training, the model should be accurate enough to be able to detect and track every single person entering and leaving the building
This people counter implementation uses object detection at fixed frame intervals and uses object tracking the rest of the time. The whole project is implemented in Pyton 3.




## Object Detection
	
The object detection is done by using YOLO_v3 implemented with openCV2. Normally Yolo v3 is very fast when implemented through darknet (Approximately 30fps), but when implemented using opencv it goes down to a mere 3-5fps. Hence object tracking has been added to speed things up a bit.

## Object Tracking
	
Object tracking here is done using dlib library which keeps track of the objects in the frame by calculating the distances of new estimated positions of the objects(estimated from previous frame) from the positions in the previous frame and saving the same id for the one having minimum distances.


## What you need
	
In order to have the code up and running, the following need to be installed
- OpenCV2
- numpy
- scipy
- dlib
- imutils
- pandas
- cmake
##### This can be done by running the command below since all requirements have been added to a requirements.txt file
    pip install -r requirements.txt
    

# After all libraries are installed,the code can now be run.
##### Run the code below to input a video
    python counter.py --video [full path of the video file]
###### Below is an example of running the code with a local video
    python counter.py --video videos/10.avi
##### If you want to use web cam : 
    python counter.py  


	

 
