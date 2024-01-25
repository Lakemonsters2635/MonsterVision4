import json
import math
from pathlib import Path
import sys
import time
import depthai as dai
import cv2

INCHES_PER_MILLIMETER = 39.37 / 1000

# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 0.4
depthWeight = 0.6


def updateBlendWeights(percent_rgb):
    """
    Update the rgb and depth weights used to blend depth/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb)/100.0
    depthWeight = 1.0 - rgbWeight



def _average_depth_coord(pt1, pt2, padding_factor):
    factor = 1 - padding_factor
    x_shift = (pt2[0] - pt1[0]) * factor / 2
    y_shift = (pt2[1] - pt1[1]) * factor / 2
    av_pt1 = (pt1[0] + x_shift), (pt1[1] + y_shift)
    av_pt2 = (pt2[0] - x_shift), (pt2[1] - y_shift)
    return av_pt1, av_pt2


class OAK:
 
    def __init__(self, devInfo : dai.DeviceInfo, laserProjectorNotUsed=None, useNN=False):
        self.rgbResolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P # Get camera resolution
        self.rgbWidth = 1920 # RGB Camera width
        self.rgbHeight = 1080 # RGB Camera height
        if useNN:                 # I think these lines set the scale of the image or change the resolution directly from the sensor
            self.ispScale = (2, 3)
        else:
            self.ispScale = (1, 6)

        self.bbfraction = 0.2 # The size of the inner bounding box as a fraction of the original

        self.CAMERA_FPS = 25
        self.DESIRED_FPS = 10		# seem to actually get 1/2 this.  Don't know why.....
        self.PREVIEW_WIDTH = 200
        self.PREVIEW_HEIGHT = 200

        self.syncNN = True
        self.devInfo = devInfo

        self.useNN = useNN

        return


    NN_FILE = "/boot/nn.json" # The file with the NN config in it

    openvinoVersions = dai.OpenVINO.getVersions()            # These lines set the openvino version from ?all versions?
    openvinoVersionMap = {}
    for v in openvinoVersions:
        openvinoVersionMap[dai.OpenVINO.getVersionName(v)] = v

    def parse_error(self, mess):
        """Report parse error."""
        print("config error in '" + self.NN_FILE + "': " + mess, file=sys.stderr)

    def read_nn_config(self):
        try: # Try to open the NN config file
            with open(self.NN_FILE, "rt", encoding="utf-8") as f:
                j = json.load(f) # Load in json format
        except OSError as err: # If file doesn't exist then throw an error
            print("could not open '{}': {}".format(self.NN_FILE, err), file=sys.stderr)
            return {}

        # top level must be an object
        if not isinstance(j, dict):
            self.parse_error("must be JSON object")
            return {}

        return j # Return the config json
    
    def setupSDN(self, nnConfig):
        # Create pipeline
        self.pipeline = dai.Pipeline()

        if not self.useNN:
            return None

        nnJSON = self.read_nn_config()
        self.LABELS = nnJSON['mappings']['labels'] # Get all possible labels from config
        nnConfig = nnJSON['nn_config'] # Get 9uspecific data to do with the NN
    
        # Get path to blob
        blob = nnConfig['blob'] # Get name of the NN blob file
        nnBlobPath = str((Path(__file__).parent / Path('models/' + blob)).resolve().absolute())

        if not Path(nnBlobPath).exists():                                # If the file doesn't exist then throw an error
            import sys

            raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

        # NN label texts
        self.labelMap = self.LABELS

        try: # Force a specific version of openvino
            self.openvinoVersion = nnConfig['openvino_version']
        except KeyError:
            self.openvinoVersion = ''

        if self.openvinoVersion != '':
            self.pipeline.setOpenVINOVersion(self.openvinoVersionMap[self.openvinoVersion])

        try: # Get the size of the input from the config file
            self.inputSize = tuple(map(int, nnConfig.get("input_size").split('x')))
        except KeyError: # Default to 300x300
            self.inputSize = (300, 300)

        family = nnConfig['NN_family'] # Get the NN family from the config
        if family == 'mobilenet':
            detectionNodeType = dai.node.MobileNetDetectionNetwork
        elif family == 'YOLO':
            detectionNodeType = dai.node.YoloDetectionNetwork
        else:
            raise Exception(f'Unknown NN_family: {family}')

        try:
            self.bbfraction = nnConfig['bb_fraction']
        except KeyError:
            self.bbfraction = self.bbfraction			# No change from default



        # Create the spatial detection network node - either MobileNet or YOLO (from above)

        spatialDetectionNetwork = self.pipeline.create(detectionNodeType)

        # Set the NN-specific stuff

        if family == 'YOLO':
            spatialDetectionNetwork.setNumClasses(nnConfig['NN_specific_metadata']['classes'])
            spatialDetectionNetwork.setCoordinateSize(nnConfig['NN_specific_metadata']['coordinates'])
            spatialDetectionNetwork.setAnchors(nnConfig['NN_specific_metadata']['anchors'])
            spatialDetectionNetwork.setAnchorMasks(nnConfig['NN_specific_metadata']['anchor_masks'])
            spatialDetectionNetwork.setIouThreshold(nnConfig['NN_specific_metadata']['iou_threshold'])
            x = nnConfig['NN_specific_metadata']['confidence_threshold']
            spatialDetectionNetwork.setConfidenceThreshold(x)
        else:
            x = nnConfig['confidence_threshold']
            spatialDetectionNetwork.setConfidenceThreshold(x)
        
        spatialDetectionNetwork.setBlobPath(nnBlobPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5) # Don't know why this is hardcoded because it's in the config file
        spatialDetectionNetwork.input.setBlocking(False) # Tells the system to overwrite the oldest data instead of waiting for the queue to clear up a bit

        return spatialDetectionNetwork # Return a NN pipeline node

    def buildDebugPipeline(self): # Add debug stuff to pipeline
        return
        

    def displayDebug(self, device): # Add debug stuff to displayed image
        return

    def buildPipeline(self, spatialDetectionNetwork):

        # Define sources and outputs

        self.camRgb = self.pipeline.create(dai.node.ColorCamera) # Create the RGB camera node

        self.xoutRgb = self.pipeline.create(dai.node.XLinkOut) # Create an output to raspberry pi node for the image
        self.xoutRgb.setStreamName("rgb") # Name it "rgb" so it can be found by the pi

        if self.useNN: # If you have an NN then create a output for the detections
            self.xoutNN = self.pipeline.create(dai.node.XLinkOut)
            self.xoutNN.setStreamName("detections")

        # Properties

        if self.useNN: # Set the preview size of the output image
            self.camRgb.setPreviewSize(self.inputSize)                    # General camera setting output stuff
        self.camRgb.setResolution(self.rgbResolution)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.camRgb.setFps(self.CAMERA_FPS)
        self.camRgb.setIspScale(self.ispScale[0], self.ispScale[1])

        # On 2023 robot, gripper camera is mounted inverted.
        
        self.camRgb.setImageOrientation(dai.CameraImageOrientation.NORMAL) # Got changed back to normal

        print("Camera FPS: {}".format(self.camRgb.getFps())) # Print the FPS

        # For now, RGB needs fixed focus to properly align with depth.
        # This value was used during calibration
        
        try:
            ovv = self.openvinoVersionMap[self.openvinoVersion]
            calibData = dai.Device(ovv, self.devInfo, False).readCalibration2()
            lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
            if lensPosition:
                self.camRgb.initialControl.setManualFocus(lensPosition)
        except Exception:
           pass 

        # Linking

        if self.useNN:
            self.camRgb.preview.link(spatialDetectionNetwork.input) # Link camera's preview output to the input of the NN node
            spatialDetectionNetwork.out.link(self.xoutNN.input) # Link NN output to the xLink detections output node
            spatialDetectionNetwork.passthrough.link(self.xoutRgb.input) # Passthrough the camera image to be displayed to the rgb xLink node
        else:
            self.camRgb.isp.link(self.xoutRgb.input) # If not using a NN then link the camera output directly to the xLink rgb output node

        # Extra for debugging

        self.buildDebugPipeline()



    def runPipeline(self, processDetections, objectsCallback=None, displayResults=None, processImages=None, cam="", imagesParam=None):
        # Connect to device and start pipeline
        with dai.Device(self.pipeline, self.devInfo) as device:

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            startTime = time.monotonic() # Frame rate timing stuff
            counter = 0
            self.fps = 0

            if self.useNN:
                detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False) # Get the NN data from the queue

                while True:
                    self.inPreview = rgbQueue.get() # Get the current frame
                    self.inDet = detectionNNQueue.get() # Get the current detections from the frame

                    counter += 1                                      # Frame rate calculations stuff
                    current_time = time.monotonic()
                    if (current_time - startTime) > 1:
                        self.fps = counter / (current_time - startTime)
                        counter = 0
                        startTime = current_time

                    self.frame = self.inPreview.getCvFrame() # Make the current rgb frame into a CVFrame ?for speed?

                    detections = self.inDet.detections # Get the list of detections from the current detections ?object?
                    if len(detections) != 0: # If there is detections then process them via the processDetections callback function
                        objects = processDetections(self, detections)
                        if objects is None: # If you process none then return none
                            objects = []
                    else:
                        objects = []

                    # if processImages is not None:
                    #     additionalObjects = processImages(self.frame, None, None, self.frame, imagesParam)
                    #     if additionalObjects is not None:
                    #         objects.extend(additionalObjects)

                    if objectsCallback is not None: # If you got a callback call it back
                        objectsCallback(objects, cam)

                    self.displayDebug(device) # Add debug info to the display

                    if displayResults is not None:
                        if displayResults(self.frame, None, None, cam) == False:
                            return
            else:
                while True:
                    self.inRgb = rgbQueue.get()
                    self.frame = self.inRgb.getCvFrame()

                    counter += 1
                    current_time = time.monotonic()
                    if (current_time - startTime) > 1:
                        self.fps = counter / (current_time - startTime)
                        counter = 0
                        startTime = current_time

                    objects = []

                    if processImages is not None:
                        additionalObjects = processImages(self.frame, None, None, self.frame, imagesParam)
                        if additionalObjects is not None:
                            objects = objects + additionalObjects

                    if objectsCallback is not None:
                        objectsCallback(objects, cam)

                    self.displayDebug(device)

                    if displayResults is not None:
                        if displayResults(self.frame, None, None, cam) == False:
                            return


    def processDetections(self, detections):
        # If the frame is available, draw bounding boxes on it and show the frame
        height = self.frame.shape[0]
        width = self.frame.shape[1]

        # re-initializes objects to zero/empty before each frame is read
        objects = []

        for detection in detections:
            # Find center of bounding box

            cX = (detection.xmin + detection.xmax) / 2
            cY = (detection.ymin + detection.ymax) / 2
            R = max((detection.xmax - detection.xmin)*width, (detection.ymax - detection.ymin)*height) / (2*width)

            # Denormalize bounding box.  Coordinates in pixels on frame

            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)

            try:
                label = self.labelMap[detection.label]

            except KeyError:
                label = detection.label

            if detection.label == 1:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            #print(detection.spatialCoordinates.x, detection.spatialCoordinates.y, detection.spatialCoordinates.z)

            cv2.putText(self.frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(self.frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(self.frame, f"X: {round(cX, 3)} in", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(self.frame, f"Y: {round(cY,3)} in", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(self.frame, f"R: {round(R, 3)} in", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            cv2.circle(self.frame, (int(cX*width), int(cY*height)), 5, (0, 0, 255), -1)
            cv2.circle(self.frame, (int(cX*width), int(cY*height)), int(R*width), (255, 0, 0), 2)

            # cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

            objects.append({"objectLabel": self.LABELS[detection.label], "x": cX,
                            "y": cY, "z": R,
                            "confidence": round(detection.confidence, 2)})

        cv2.putText(self.frame, "NN fps: {:.2f}".format(self.fps), (2, self.frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                    (255, 255, 255))

        return objects                
