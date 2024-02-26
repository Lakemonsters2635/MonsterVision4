import cv2
import queue

cscoreAvailable = True
try:
    from cscore import CameraServer
except ImportError:
    cscoreAvailable = False

class MTD:
    def __init__(self, previewWidth, previewHeight):
        self.Q = queue.Queue()
        self.allDone = False
        
        if cscoreAvailable:
            # self.cs = CameraServer.getInstance()
            CameraServer.enableLogging()
            self.csoutput = CameraServer.putVideo("MonsterVision", previewWidth, previewHeight) # TODOnot        


    def enqueueImage(self, window : str, image):
        self.Q.put((window, image))

    def displayLoop(self):
        while True:
            (window, image) = self.Q.get(True) # Get the image from the queue window named  "DS Image"
            # If it's the right window name and cscore is available then put it onto the camera server website
            if window == "DS Image":
                if cscoreAvailable:
                    self.csoutput.putFrame(image)
                continue
            cv2.imshow(window, image) # Show the image that you output to the server to the screen
            self.Q.task_done()
            wk = cv2.waitKey(1)
            self.allDone = self.allDone or wk == 113
            if self.allDone: 
                print("Done")
                return
        return

    def areWeDone(self):
        return self.allDone

