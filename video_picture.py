# Importing all necessary libraries
import cv2
import os
import time

# Read the video from specified path
cam = cv2.VideoCapture("videos/29.avi")

try:

    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0
skipFrames = 15


while (True):
    # time.sleep(5) # take schreenshot every 5 seconds
    # reading from frame
    ret, frame = cam.read()

    if currentframe % skipFrames == 0:
        # if video is still left continue creating images
        name = './data/frame' + str(currentframe) + '.jpg'
        print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        
    else:
        pass
    
    currentframe += 1

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
