# Import libraries
import json
import sys
import time

usingNTCore = False
try:
# Older OSes use pynetworktables
    from networktables import NetworkTables
    from networktables import NetworkTablesInstance
except ImportError:
# New have ntcore preinstalled
    import ntcore
    usingNTCore = True

import cv2
import MultiThreadedDisplay4 as MTD
import threading
import platform

# Something with cscore which we don't use in this file
cscoreAvailable = True
try:
    from cscore import CameraServer
except ImportError:
    cscoreAvailable = False

# Set camera FPS
CAMERA_FPS = 25
DESIRED_FPS = 10		# seem to actually get 1/2 this.  Don't know why.....; THIS IS BECAUSE OF MATH BELOW MEANS THAT YOU GET 25  % 10 = 5


class FRC:
    ROMI_FILE = "/boot/romi.json"   # idk
    FRC_FILE = "/boot/frc.json"     # Some camera settings incuding laser power
    NN_FILE = "/boot/nn.json"       # NN config file


    def __init__(self, previewWidth, previewHeight):
        # Tells you if you are on the robot or not by looking at the platform name (if you are using the WPILib pi image?)
        # onRobot really should be called "headless".  It means there's no graphics capability on the underlying hardware

        self.onRobot = platform.uname().node == "wpilibpi"

       # Team number
        self.team = 0 # 2635
        # If the pi is setup as a sever or a client
        self.server = False
        # If a display is connected
        self.hasDisplay = False # True for testing; False in real one
        # NetworkTable Instance holder; Initialized below
        self.ntinst = None
        # Vision NetworkTable; Initialized below; getTable MonsterVision
        self.sd = None
        # Num frames; Maybe used for FPS counting?
        self.frame_counter = 0
        # Current of the Laser Projector on OAK-D pro; Optimum is 765.0 (mA)
        self.LaserDotProjectorCurrent = 0
        # FPS counting
        self.lastTime = 0

        self.mtd = MTD.MTD(previewWidth, previewHeight) # I think image put onto WPILibPi website for driver viewing
        self.lockNT = threading.Lock() # Lock onto NetworkTable thread for posting detections later

        # Probably frame storage for later
        self.aprilImage = None
        self.detectionsImage = None

        self.read_frc_config() # Read the FRC config file and initialize above variables

        if usingNTCore:
            self.ntinst = ntcore.NetworkTableInstance.getDefault()
        else:
            self.ntinst = NetworkTablesInstance.getDefault() # Create a NetworkTable Instance

        # Sets up the NT depending on config
        if self.server:
            print("Setting up NetworkTables server")
            self.ntinst.startServer()
        else:
            print("Setting up NetworkTables client for team {}".format(self.team))
            self.ntinst.startClientTeam(self.team)
            self.ntinst.startDSClient()

        if usingNTCore:
            self.sd = self.ntinst.getTable("MonsterVision")
        else:
            self.sd = NetworkTables.getTable("MonsterVision") # Get the MonsterVision NT; Maybe creates it


    # Return True if we're running on Romi.  False if we're a coprocessor on a big 'bot
    # Never used but checks if the files exists
    def is_romi(self):
        try:
            with open(self.ROMI_FILE, "rt", encoding="utf-8") as f:
                json.load(f)
                # j = json.load(f)
        except OSError as err:
            print("Could not open '{}': {}".format(self.ROMI_FILE, err), file=sys.stderr)
            return False
        return True

    # Never used but checks if the files exists
    def is_frc(self):
        try:
            with open(self.FRC_FILE, "rt", encoding="utf-8") as f:
                json.load(f)
        except OSError as err:
            print("Could not open '{}': {}".format(self.FRC_FILE, err), file=sys.stderr)
            return False
        return True

    def parse_error(self, mess):
        """Report parse error."""
        print("config error in '" + self.FRC_FILE + "': " + mess, file=sys.stderr)

    # Read config file
    def read_frc_config(self):
        # Try to open it and then stores it as a json object in variable j
        try:
            with open(self.FRC_FILE, "rt", encoding="utf-8") as f:
                j = json.load(f)
        except OSError as err:
            print("could not open '{}': {}".format(self.FRC_FILE, err), file=sys.stderr)
            return False

        # top level must be an object
        if not isinstance(j, dict):
            self.parse_error("must be JSON object")
            return False

        # Is there an desktop display?
        try:
            self.hasDisplay = j["hasDisplay"]
        except KeyError:
            self.hasDisplay = False

        # Sets team number
        try:
            self.team = j["team"]
        except KeyError:
            self.parse_error("could not read team number")
            return False

        # ntmode (optional)
        # Sets NTmode as client or server based on config file
        if "ntmode" in j:
            s = j["ntmode"]
            if s.lower() == "client":
                self.server = False
            elif s.lower() == "server":
                self.server = True
            else:
                self.parse_error(f"could not understand ntmode value '{s}'")

        # Sets LaserDotProjectorCurrent in mA
        try:
            self.LaserDotProjectorCurrent = j["LaserDotProjectorCurrent"]
        except KeyError:
            self.LaserDotProjectorCurrent = 0

        self.LaserDotProjectorCurrent *= 1.0
        
        return True
    
    # NT writing for NN detections and AprilTags
    def writeObjectsToNetworkTable(self, jsonObjects, cam):
        # Protect NT access with a lock.  Just in case NT implementation is not thread-safe
        self.lockNT.acquire() # Acquire the NT's thread and block other attempts to lock-on/acquire
        # Put onto table with the ID of ex."ObjectTracker-Chassis" and overwrites the previous frame data
        # Looks like this on the table: "[{'objectLabel': 'cone', 'x': 2, 'y': -3, 'z': 27, 'confidence': 0.87}]"
        self.sd.putString("ObjectTracker-" + cam, jsonObjects)
        self.ntinst.flush() # Puts all values onto table immediately
        self.lockNT.release() # Release the thread to allow other locks or whatever you want

    def displayResults(self, fullFrame, depthFrameColor, detectionFrame, cam):
        if self.hasDisplay: # If you have a display create a bunch of display windows
            # FPS counting and put onto OpenCV frame
            now = time.monotonic_ns()
            fps = 1000000000.0 / (now - self.lastTime)
            self.lastTime = now
            cv2.putText(fullFrame, "NN fps: {:.2f}".format(fps), (2, fullFrame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
            (255, 255, 255))

            # These create the pop-up display windows for each
            # We don't use cv2 because it doesn't work with multiple threads
            if depthFrameColor is not None:
                self.mtd.enqueueImage("depth " + cam, depthFrameColor)
                # cv2.imshow("depth " + cam, depthFrameColor)
            if detectionFrame is not None:
                self.mtd.enqueueImage("detections " + cam, detectionFrame)
                # cv2.imshow("detections " + cam, detectionFrame)
            if fullFrame is not None:
                self.mtd.enqueueImage("openCV " + cam, fullFrame)
                # cv2.imshow("openCV " + cam, fullFrame)

        # Depending on the camera put a different into each variable; We merge later
        if cam == "AprilTagPro":
            self.aprilImage = fullFrame
            # cv2.rectangle(self.aprilImage, (40, 0), (270, 128), (255, 255, 255), 4)
            # cv2.ellipse(self.aprilImage, (150, 30), (80, 70), 0, 0, 360, (255,255,255), 4)
        else:
            self.detectionsImage = detectionFrame
        
        # Every DESIRED_FPS frame output to driver station/website view a merge of both image frames
        # if self.frame_counter % (CAMERA_FPS / DESIRED_FPS) == 0: # THIS IS THE MATH THAT ONLY GETS YOU HALF OF YOUR DESIRED FPS
        if True:
            # If you have both images then do a combine
            if self.aprilImage is not None and self.detectionsImage is not None:
                img = cv2.hconcat([self.aprilImage, self.detectionsImage]) # Merges Frames together
            # If you only have the AprilTag one then output just that one
            elif self.aprilImage is not None:
                img = self.aprilImage
            # Otherwise if you have no image or just detections image then output that one
            else:
                img = self.detectionsImage

            # If you aren't on a robot then pop-up one window of what the driver station view should be
            if not self.onRobot:
                self.mtd.enqueueImage("DS View", img)
            self.mtd.enqueueImage("DS Image", img)              # Special window name causes MTD to send to camera server
            

        self.frame_counter += 1

        # if self.hasDisplay and cv2.waitKey(1) == ord('q'):
        #     return False
        if self.mtd.areWeDone():
            return False
        
        return True

    def runDisplay(self):
        self.mtd.displayLoop()
        return
