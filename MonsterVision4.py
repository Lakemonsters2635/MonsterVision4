#!/usr/bin/env python3

from ast import List
import json
import threading
import time
import FRC4
import AprilTags4
import cv2
import importlib
import Contours4 
import DAI4
import depthai as dai
import RANSAC4

# plane = RANSAC.findPlane((-3, -5, 15), (0, 0, 15), (3, -5, 15))
# print(plane)
# plane = RANSAC.findPlane((-3, -5, 18), (0, 0, 15), (3, -5, 12))
# print(plane)
# plane = RANSAC.findPlane((-3, -5, 16.73205), (0, 0, 15), (3, -5, 13.26795))
# print(plane)

def processExtra1(imageFrame, depthFrame, depthFrameColor, drawingFrame, contours):
    return contours.detect(imageFrame, depthFrame, depthFrameColor, drawingFrame)

def processExtraD(imageFrame, depthFrame, depthFrameColor, drawingFrame, aprilTags):
    return aprilTags.detect(imageFrame, depthFrame, depthFrameColor, drawingFrame)

def processDetections(oak, detections):
    return oak.processDetections(detections)

def objectsCallback(objects, cam):
    global frc
    frc.writeObjectsToNetworkTable(json.dumps(objects), cam)

def displayResults(fullFrame, depthFrameColor, detectionFrame, cam):
    return frc.displayResults(fullFrame, depthFrameColor, detectionFrame, cam)




def runOAK1(devInfo, cam):
    try:
        OAK = importlib.import_module("Gripper4")            # Allows substitution of other pilelines!
        contours = Contours4.Contours()
        # Setup the oak-1 camera by creating an OAK object from the Gripper module with the camera info, the presence of a laser projector, and whether or not to use NN
        oak = OAK.OAK(devInfo, None, useNN=True)
        nnConfig = oak.read_nn_config() # Call function of the OAK object to read NN config

        spatialDetectionNetwork = oak.setupSDN(nnConfig) # Call function of the OAK object to setup SDN
        oak.buildPipeline(spatialDetectionNetwork) # Call function of the OAK object to setup pipeline

        oak.runPipeline(processDetections, objectsCallback, displayResults, processExtra1, cam, contours) # Run the camera
    except:
        print("runOAK1 died.  Killing MonsterVision3")
        quit()                                          # If any exception occurs, quit the entire program.
                                                        # The WPI infrastructure will restart us
    return

def runOAKD(devInfo, cam):
    try:
        OAK = importlib.import_module("MV3")            # Allows substitution of other pilelines!
        aprilTags = AprilTags4.AprilTags("tag16h5")
        oak = OAK.OAK(devInfo, frc.LaserDotProjectorCurrent)
        nnConfig = oak.read_nn_config()

        spatialDetectionNetwork = oak.setupSDN(nnConfig)
        oak.buildPipeline(spatialDetectionNetwork)

        oak.runPipeline(processDetections, objectsCallback, displayResults, processExtraD, cam, aprilTags)
    except:
        print("runOAKD died.  Killing MonsterVision3")
        quit()                                          # If any exception occurs, quit the entire program.
                                                        # The WPI infrastructure will restart us
    return

PREVIEW_WIDTH = 200
PREVIEW_HEIGHT = 200

frc = FRC4.FRC(PREVIEW_WIDTH, PREVIEW_HEIGHT)

OAK_D_MXID = None
OAK_1_MXID = None

# devices = DAI.DAI.getDevices()

# for c in devices:
#     if c["cameras"] == 1:
#         if OAK_1_MXID is None:
#             OAK_1_MXID = c["mxid"]
#         else:
#             print(f"Found multiple OAK-1 devices.  Using {OAK_1_MXID}")
#     elif c["cameras"] == 3:
#         if OAK_D_MXID is None:
#             OAK_D_MXID = c["mxid"]
#         else:
#             print(f"Found multiple OAK-D devices.  Using {OAK_D_MXID}")
#     else:
#         print(f'Found device {c["mxid"]} having {c["cameras"]}.  This is unusual.')

infos = dai.DeviceBootloader.getAllAvailableDevices()
# OAK_D_MXID = "14442C105129C6D200"     # Original OAK-D at Michael's house
OAK_1_MXID = "14442C10E1474FD000"
OAK_D_MXID = '1944301001564D1300'       # OAK-D Pro

def checkCam(infos, mxid):
    for i in infos:
        if mxid == i.mxid: return i
    return None

OAK_D_DEVINFO = checkCam(infos, OAK_D_MXID)
OAK_1_DEVINFO = checkCam(infos, OAK_1_MXID)

if OAK_D_DEVINFO is None and OAK_1_DEVINFO is None:
    print("No cameras found")
    exit()

print("Using cameras:")
if OAK_D_DEVINFO is not None: print(f"{OAK_D_MXID} OAK-D")
if OAK_1_DEVINFO is not None: print(f"{OAK_1_MXID} OAK-1")

thread1 = None
threadD = None

if OAK_1_DEVINFO is not None:
    thread1 = threading.Thread(target=runOAK1, args=(OAK_1_DEVINFO, "Gripper", ))
    thread1.start()

if OAK_D_DEVINFO is not None:
    threadD = threading.Thread(target=runOAKD, args=(OAK_D_DEVINFO, "Chassis", ))
    threadD.start()

# Now call the display queue worker loop (must be run from main thread)
# This should never return until the user types 'q' (if windows are being used)

frc.runDisplay()

print("Main thread died.  Killing MonsterVision3")
quit()                              # Should never get here.  If we do, kill everything
                                    # Let the WPI infrastructure restart us.

