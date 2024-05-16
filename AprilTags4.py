import random
import time

usingRobotPyAprilTag = False
try:
    import apriltag
except ImportError:
    import robotpy_apriltag as apriltag
    usingRobotPyAprilTag = True

import cv2
import math
import numpy as np
import RANSAC4
import logging
import depthai as dai
import PLANE_FIT4

INCHES_PER_MILLIMETER = 39.37 / 1000
RANSAC_SCALING_FACTOR = 10.0 / 6.0              # Size of AprilTag background / tag size

class AprilTags:
    # Required information for calculating spatial coordinates on the host
    monoFOV = np.deg2rad(72)
    tanHalfHFOV = math.tan(monoFOV / 2.0)

    penucheFactorM = .8047
    penucheFactorB = 5.0061 * 25.4
    # penucheFactorM = 0.785
    # penucheFactorB = 10 * 25.4

    def calc_tan_angle(self, offset, depthWidth):
        return offset * self.tanHalfHFOV / depthWidth

    def mapXCoord(self, Small):
        return int(Small * self.Scale + self.xMin)

    def mapYCoord(self, Small):
        return int(Small * self.Scale + self.yMin)

    def mapCoords(self, pt):
        return (self.mapXCoord(pt[0]), self.mapYCoord(pt[1]))

    #@profile
    def findPlane(self, p1, p2, p3):

# Find the plane's normal vector by taking the cross product of two vectors between pairs of points
        (P10, P11, P12) = p1
        (P20, P21, P22) = p2
        (P30, P31, P32) = p3

        (a0, a1, a2) = (P20-P10, P21-P11, P22-P12)
        (b0, b1, b2) = (P30-P10, P31-P11, P32-P12)
        (N0, N1, N2) = (a1*b2-a2*b1, a2*b0-a0*b2, a0*b1-a1*b0)

# Pick any point to compute D

        D = N0*P10 + N1*P11 + N2*P12
        U = math.sqrt(N0*N0+N1*N1+N2+N2)

# Plane in Ax + By + Cz = D form, where (A, B, C) is the unit vector normal to the plane
        return (N0/U, N1/U, N2/U, D/U)


    @staticmethod
    def poseAngleFromVector(x, y):
        angle = math.pi - math.atan2(y, x)
        return angle


    #@profile
    def getPoseAngles(self, depth, xmin, xmax, ymin, ymax, inputShape): # (depthFrame, xmin, xmax, ymin, ymax, inputShape)
        pointCount = (xmax-xmin) * (ymax-ymin)
        pointCloud = np.full((pointCount, 3), (0.0, 0.0, 0.0))

# Create the point cloud from the depth data

        # make sure we don't have too many points sampled.  Stride the data to sample
        # the available points when creating our point cloud for making the plane
        # ensure stride is at least 1
        stride = int(((xmax - xmin) * (ymax - ymin) / 400) ** 0.5)
        stride = 1 if stride < 1 else stride

        index = 0
        yLen = int((ymax - ymin) / stride)

        for x in range(xmin, xmax, stride):
            try:
                tanAngle_x = self.calc_tan_angle(x - int(depth.shape[1] / 2), inputShape)

                zv = np.array(depth[ymin:ymax:stride, x]) * self.penucheFactorM + self.penucheFactorB
                zmask = zv != self.penucheFactorB

                yv = (np.arange(ymin, ymax, stride) - int(depth.shape[0] / 2)) * self.tanHalfHFOV / inputShape

                pc = np.array(list(zip(zv*tanAngle_x, -zv*yv, zv)))
                num = zmask.sum()

                pointCloud[index:index+num] = pc[zmask]
                index += num
            except:
                # line83 periodically fails due to an error similar to:
                #    operands can not be broadcast together with shapes (12,) and (13,)
                # should come back and figure out what is going on.
                pass

        # PLANE_FIT4.FIT_PLANE(pointCloud, pointCount)
        plane = RANSAC4.RANSAC(pointCloud[0:index], index)

        if plane is None:
            return (None, None)

# Find pose of tag.  In particular, find its rotation about its X and Y axes.

        (A, B, C, D) = plane
          
        xAngle = math.pi - math.atan2(C, B)
        # if xAngle is not None:
            # print("A: {0:.2f}  C: {1:.2f}  X:{2:.2f}".format(A, C, xAngle*180/math.pi)) 
                  
        yAngle = math.pi - math.atan2(C, A)
        # if yAngle is not None:
        #     print("A: {0:.2f}  B: {1:.2f}  Y: {2:.2f}".format(A, B, yAngle*180/math.pi)) 
                
        return (xAngle, yAngle)
    
    # This code assumes depth symmetry around the centroid

    # Calculate spatial coordinates from depth map and bounding box (ROI)
    #@profile
    def calc_spatials(self, bbox, centroidX, centroidY, depth, averaging_method=np.median): # ((ptA[0], ptA[1], ptC[0], ptC[1]), cX, cY, depthFrame)
        if depth is None:
            return (centroidX, centroidY, 0)
        
        inputShape = depth.shape[0]
        xmin, ymin, xmax, ymax = bbox
        if xmin > xmax:  xmin, xmax = xmax, xmin
        if ymin > ymax:  ymin, ymax = ymax, ymin

        if xmin == xmax or ymin == ymax: # Box of size zero
            return None

        # Calculate the average depth in the ROI.
        depthROI = depth[ymin:ymax, xmin:xmax]        
        averageDepth = averaging_method(depthROI)

        bb_x_pos = centroidX - int(depth.shape[1] / 2)
        bb_y_pos = centroidY - int(depth.shape[0] / 2)

        tanAngle_x = self.calc_tan_angle(bb_x_pos, inputShape)
        tanAngle_y = self.calc_tan_angle(bb_y_pos, inputShape)

        z = averageDepth * self.penucheFactorM + self.penucheFactorB

        x = z * tanAngle_x
        y = -z * tanAngle_y

# Because the lower arm of the 2023 robot cuts off a little bit of the image, we cannot expand
# the region usedfor angle.

# # The AprilTags are 6" square.  They are printed on a 10" square.  We can use the extra area to make the
# # pose angle calculation more accurate.

#         width = xmax - xmin
#         height = ymax - ymin

#         xShift = int(width*(RANSAC_SCALING_FACTOR-1.0)/2)
#         yShift = int(height*(RANSAC_SCALING_FACTOR-1.0)/2)

#         xmin1 = max(0, xmin - xShift)
#         xmax1 = min(depth.shape[1]-1, xmax + xShift)
#         ymin1 = max(0, ymin - yShift)
#         ymax1 = min(depth.shape[0]-1, ymax+  yShift)

#         (xAngle, yAngle) = self.getPoseAngles(depth, xmin1, xmax1, ymin1, ymax1, inputShape)
        (xAngle, yAngle) = self.getPoseAngles(depth, xmin, xmax, ymin, ymax, inputShape)

        return (x,y,z, xAngle, yAngle)

    # def getTrapAngles()
        
    def __init__(self, tagFamilies, cameraIntrinsics=None, rgbHFOV = None): # Never use the cameraIntrinsics variable and it was throwing an error so I default it to None

        self.tagFamily = tagFamilies
        self.lastFrameTime = time.time_ns() / 1.0e9

        if usingRobotPyAprilTag:
            self.detector = apriltag.AprilTagDetector()
            self.detector.addFamily(tagFamilies)
        else:
            options = apriltag.DetectorOptions(families=tagFamilies)
            self.detector = apriltag.Detector(options)
        
        # with 

        self.intrinsics = cameraIntrinsics
        print(f"Camera Intrinsics: {cameraIntrinsics}")
        if rgbHFOV is not None:
            self.rgbHFOV = np.deg2rad(rgbHFOV)
            self.tanHalfHFOV = math.tan(self.rgbHFOV / 2.0)
       
    #@profile
    def drawBoundingBox(self, frame, ptA, ptB, ptC, ptD, color, lineWidth):
        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (255, 0, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 0, 255), 2)
        cv2.line(frame, ptD, ptA, (255, 255, 255), 2)


    #@profile
    def detect(self, imageFrame, depthFrame, depthFrameColor, drawingFrame):

        objects = []
        now = time.time_ns() / 1.0e9
        fps = 1/(now - self.lastFrameTime)
        self.lastFrameTime = now

        gray = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(gray)

        if drawingFrame is not None:
            dfw = drawingFrame.shape[1]
            dfh = drawingFrame.shape[0]
            ifw = imageFrame.shape[1]
            ifh = imageFrame.shape[0]

            xScale = dfw / ifw
            yScale = dfh / ifh

            if xScale < yScale:
                self.Scale = yScale
                self.xMin = int(dfw/2 - ifw*self.Scale/2)
                self.yMin = 0
            else:
                self.Scale = xScale
                self.yMin = int(dfh/2 - ifh*self.Scale/2)
                self.xMin = 0

        # loop over the AprilTag detection results
        for r in results:
            if not usingRobotPyAprilTag and r.hamming != 0:
                continue





            # TESTING GET POSE LIB
            # print(self.intrinsics)
            # print(self.detector.detection_pose(r, self.intrinsics))





            # extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            if usingRobotPyAprilTag:
                ptA = r.getCorner(0)
                ptB = r.getCorner(1)
                ptC = r.getCorner(2)
                ptD = r.getCorner(3)
                ptA = (int(ptA.x), int(ptA.y))
                ptB = (int(ptB.x), int(ptB.y))
                ptC = (int(ptC.x), int(ptC.y))
                ptD = (int(ptD.x), int(ptD.y))
                center = r.getCenter()
                cX = int(center.x)
                cY = int(center.y)
                res = self.calc_spatials((ptA[0], ptA[1], ptC[0], ptC[1]), int(center.x), int(center.y), depthFrame)
            else:
                (ptA, ptB, ptC, ptD) = r.corners
                ptB = (int(ptB[0]), int(ptB[1]))
                ptC = (int(ptC[0]), int(ptC[1]))
                ptD = (int(ptD[0]), int(ptD[1]))
                ptA = (int(ptA[0]), int(ptA[1]))
                (cX, cY) = (int(r.center[0]), int(r.center[1]))
                res = self.calc_spatials((ptA[0], ptA[1], ptC[0], ptC[1]), cX, cY, depthFrame)
            if res == None:
                continue
            (atX, atY, atZ, xAngle, yAngle) = res
            if depthFrame is None:
                atX = atX / imageFrame.shape[1] - 0.5
                atY = atY / imageFrame.shape[0] - 0.5
                units = ""
            else:
                atX = round((atX * INCHES_PER_MILLIMETER), 1)
                atY = round((atY * INCHES_PER_MILLIMETER), 1)
                atZ = round((atZ * INCHES_PER_MILLIMETER), 1)
                units = "in"

            # draw the bounding box of the AprilTag detection
            self.drawBoundingBox(imageFrame, ptA, ptB, ptC, ptD, (0, 255, 0), 2)

            
            # draw the center (x, y)-coordinates of the AprilTag
            cv2.circle(imageFrame, (cX, cY), 5, (0, 0, 255), -1)

            wd = abs(ptC[0] - ptA[0])
            ht = abs(ptC[1] - ptA[1])
            lblX = int(cX - wd/2)
            lblY = int(cY - ht/2)
            # draw the tag family on the image
            # tagID= '{}: {}'.format(r.tag_family.decode("utf-8"), r.tag_id)
            tagID = self.tagFamily
            color = (0, 0, 255)

            if lblY < 75:
                lblY = 75
            if lblY > imageFrame.shape[0]:
                lblY = imageFrame.shape[0]

            cv2.putText(imageFrame, tagID, (lblX, lblY - 75), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(imageFrame, f" X: {atX} {units}", (lblX, lblY - 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(imageFrame, f" Y: {atY} {units}", (lblX, lblY - 45), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(imageFrame, f" Z: {atZ} {units}", (lblX, lblY - 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            if xAngle is not None:
                xAngleDeg = round(xAngle*180/math.pi, 0)
                cv2.putText(imageFrame, f"XA: {xAngleDeg} deg", (lblX, lblY - 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            if yAngle is not None:
                yAngleDeg = round(yAngle*180/math.pi, 0)
                cv2.putText(imageFrame, f"YA: {yAngleDeg} deg", (lblX, lblY + 0), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            if drawingFrame is not None:
                aa = self.mapCoords(ptA)
                bb = self.mapCoords(ptB)
                cc = self.mapCoords(ptC)
                dd = self.mapCoords(ptD)
                ctr = self.mapCoords((cX, cY))
                self.drawBoundingBox(drawingFrame, aa, bb, cc, dd, (0, 0, 0), 4)
                cv2.circle(drawingFrame, ctr, 5, (0, 0, 255), -1)
                wd = abs(cc[0] - aa[0])
                ht = abs(cc[1] - aa[1])
                lblX = int(ctr[0] - wd/2)
                lblY = int(ctr[1] - ht/2)
                if lblY < 90:
                    lblY = 90
                if lblY > drawingFrame.shape[0]:
                    lblY = drawingFrame.shape[0]
                # draw the tag family on the image
                # tagID= '{}: {}'.format(r.tag_family.decode("utf-8"), r.tag_id)  # Not supported in robotpy version
                tagID = self.tagFamily
                color = (0, 0, 255)
                cv2.putText(drawingFrame, tagID, (lblX, lblY - 75), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(drawingFrame, f"X: {atX} {units}", (lblX, lblY - 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(drawingFrame, f"Y: {atY} {units}", (lblX, lblY - 45), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(drawingFrame, f"Z: {atZ} {units}", (lblX, lblY - 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                if xAngle is not None:
                    cv2.putText(drawingFrame, f"XA: {xAngleDeg} deg", (lblX, lblY - 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                if yAngle is not None:
                    cv2.putText(drawingFrame, f"YA: {yAngleDeg} deg", (lblX, lblY + 0), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            if xAngle is None: xAngleDeg = 999
            if yAngle is None: yAngleDeg = 999

            objects.append({"objectLabel": tagID, "x": atX, "y": atY, "z": atZ, "xa": xAngleDeg, "ya": yAngleDeg, "fps": fps})

        fpsLocation = (10, imageFrame.shape[0]-15)
        cv2.putText(imageFrame, f"fps: {fps}", fpsLocation, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0))

        fpsLocation = (10, drawingFrame.shape[0]-15)
        cv2.putText(drawingFrame, f"fps: {fps}", fpsLocation, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0))

        # cv2.imshow("dbg", imageFrame)
        return objects
