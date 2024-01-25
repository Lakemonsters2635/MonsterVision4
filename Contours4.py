import time
import cv2
import numpy
import math
from enum import Enum

class Contours:

    """
    An OpenCV pipeline generated by GRIP.
    """
    
    def __init__(self):
        """initializes all values to presets or None if need to be set
        """

        self.__blur_type = BlurType.Median_Filter
        # self.__blur_type = BlurType.Gaussian_Blur
        self.__blur_radius = 1.7809910068770947

        self.blur_output = None

        self.__cone_hsv_threshold_input = self.blur_output
        self.__cone_hsv_threshold_hue = [16, 33]
        self.__cone_hsv_threshold_saturation = [167, 255]
        self.__cone_hsv_threshold_value = [0, 255]

        self.cone_hsv_threshold_output = None

        self.__cube_hsv_threshold_input = self.blur_output
        self.__cube_hsv_threshold_hue = [117, 143]
        self.__cube_hsv_threshold_saturation = [94, 220]
        self.__cube_hsv_threshold_value = [110, 255]

        self.cube_hsv_threshold_output = None

        self.__cone_find_contours_input = self.cone_hsv_threshold_output
        self.__cone_find_contours_external_only = True

        self.cone_find_contours_output = None

        self.__cube_find_contours_input = self.cube_hsv_threshold_output
        self.__cube_find_contours_external_only = True

        self.cube_find_contours_output = None


        self.__circles_min_radius = 0.1


    def process(self, source0):
        """
        Runs the pipeline and sets all outputs to new values.
        """
        self.startTime = time.monotonic_ns()
        
        # # Step Blur0:
        # self.__blur_input = source0
        # (self.blur_output) = self.__blur(self.__blur_input, self.__blur_type, self.__blur_radius)
        (self.blur_output) = source0

        self.blurTime = time.monotonic_ns()

        # Step HSV_Threshold0:
        self.__cone_hsv_threshold_input = self.blur_output
        (self.cone_hsv_threshold_output) = self.__hsv_threshold(self.__cone_hsv_threshold_input, self.__cone_hsv_threshold_hue, self.__cone_hsv_threshold_saturation, self.__cone_hsv_threshold_value)
        # cv2.imshow("threshold0", self.cone_hsv_threshold_output)

        self.hsvConeTime = time.monotonic_ns()

        # Step HSV_Threshold1:
        self.__cube_hsv_threshold_input = self.blur_output
        (self.cube_hsv_threshold_output) = self.__hsv_threshold(self.__cube_hsv_threshold_input, self.__cube_hsv_threshold_hue, self.__cube_hsv_threshold_saturation, self.__cube_hsv_threshold_value)
        # cv2.imshow("threshold1", self.cube_hsv_threshold_output)

        self.hsvCubeTime = time.monotonic_ns()

        # Step Find_Contours0:
        self.__cone_find_contours_input = self.cone_hsv_threshold_output
        (self.cone_find_contours_output) = self.__find_contours(self.__cone_find_contours_input, self.__cone_find_contours_external_only)

        self.contourConeTime = time.monotonic_ns()

        # Step Find_Contours1:
        self.__cube_find_contours_input = self.cube_hsv_threshold_output
        (self.cube_find_contours_output) = self.__find_contours(self.__cube_find_contours_input, self.__cube_find_contours_external_only)

        self.contourCubeTime = time.monotonic_ns()

        # Approximate contours to polygons + get bounding rects and circles

        cone_contours_poly = [None]*len(self.cone_find_contours_output)
        cone_boundRects = [None]*len(self.cone_find_contours_output)
        cone_centers = [None]*len(self.cone_find_contours_output)
        cone_radii = [None]*len(self.cone_find_contours_output)
        for i, c in enumerate(self.cone_find_contours_output):
            cone_contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            cone_boundRects[i] = cv2.boundingRect(cone_contours_poly[i])
            cone_centers[i], cone_radii[i] = cv2.minEnclosingCircle(cone_contours_poly[i])

        self.circleConeTime = time.monotonic_ns()

        cube_contours_poly = [None]*len(self.cube_find_contours_output)
        cube_boundRects = [None]*len(self.cube_find_contours_output)
        cube_centers = [None]*len(self.cube_find_contours_output)
        cube_radii = [None]*len(self.cube_find_contours_output)
        for i, c in enumerate(self.cube_find_contours_output):
            cube_contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            cube_boundRects[i] = cv2.boundingRect(cube_contours_poly[i])
            cube_centers[i], cube_radii[i] = cv2.minEnclosingCircle(cube_contours_poly[i])

        self.circleCubeTime = time.monotonic_ns()

        return (
            (self.cone_find_contours_output, cone_boundRects, cone_centers, cone_radii),
            (self.cube_find_contours_output, cube_boundRects, cube_centers, cube_radii)
        )


    @staticmethod
    def __blur(src, type, radius):
        """Softens an image using one of several filters.
        Args:
            src: The source mat (numpy.ndarray).
            type: The blurType to perform represented as an int.
            radius: The radius for the blur as a float.
        Returns:
            A numpy.ndarray that has been blurred.
        """
        if(type is BlurType.Box_Blur):
            ksize = int(2 * round(radius) + 1)
            return cv2.blur(src, (ksize, ksize))
        elif(type is BlurType.Gaussian_Blur):
            ksize = int(6 * round(radius) + 1)
            return cv2.GaussianBlur(src, (ksize, ksize), round(radius))
        elif(type is BlurType.Median_Filter):
            ksize = int(2 * round(radius) + 1)
            return cv2.medianBlur(src, ksize)
        else:
            return cv2.bilateralFilter(src, -1, round(radius), round(radius))

    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        """Segment an image based on hue, saturation, and value ranges.
        Args:
            input: A BGR numpy.ndarray.
            hue: A list of two numbers the are the min and max hue.
            sat: A list of two numbers the are the min and max saturation.
            lum: A list of two numbers the are the min and max value.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

    @staticmethod
    def __find_contours(input, external_only):
        """Sets the values of pixels in a binary image to their distance to the nearest black pixel.
        Args:
            input: A numpy.ndarray.
            external_only: A boolean. If true only external contours are found.
        Return:
            A list of numpy.ndarray where each one represents a contour.
        """
        if(external_only):
            mode = cv2.RETR_EXTERNAL
        else:
            mode = cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        # xyz = cv2.findContours(input, mode=mode, method=method)
        contours, hierarchy =cv2.findContours(input, mode=mode, method=method)

        return contours
    

    def __processDetections(self, imageFrame, d, type):
        objects = []
        (contours, boundRects, centers, radii) = d
        for i, c in enumerate(centers):
            r = radii[i]/imageFrame.shape[1]
            if r < self.__circles_min_radius:
                continue

            # draw the center (x, y)-coordinates of the detection
            (cX, cY) = (int(c[0]), int(c[1]))
            cv2.circle(imageFrame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.circle(imageFrame, (cX, cY), int(radii[i]), (255, 0, 0), 2)

            lblX = int(cX)
            lblY = int(cY)
            color = (255, 0, 0)

            XX = round(c[0]/imageFrame.shape[1], 3)
            YY = round(c[1]/imageFrame.shape[0], 3)
            RR = round(r, 3)

            # draw the location text
            cv2.putText(imageFrame, type, (lblX, lblY - 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(imageFrame, f"X: {XX}", (lblX, lblY - 45), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(imageFrame, f"Y: {YY}", (lblX, lblY - 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(imageFrame, f"R: {RR}", (lblX, lblY - 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            obj = { "objectLabel": f"{type}", "x": XX, "y": YY, "r": round(radii[i]/imageFrame.shape[1], 3) }
            objects.append(obj) 
        return objects


    def detect(self, imageFrame, depthFrame, depthFrameColor, frame):
        detections = self.process(imageFrame)
        # blurTime = self.blurTime - self.startTime
        # hsvConeTime = self.hsvConeTime - self.blurTime
        # hsvCubeTime = self.hsvCubeTime - self.hsvConeTime
        # contourConeTime = self.contourConeTime - self.hsvCubeTime
        # contourCubeTime = self.contourCubeTime - self.contourConeTime
        # circleConeTime = self.circleConeTime - self.contourCubeTime
        # circleCubeTime = self.circleCubeTime - self.circleConeTime

        # print(f"Blur:\t\t{blurTime}\nhsvCone:\t{hsvConeTime}\nhsvCube:\t{hsvCubeTime}\ncontourCone:\t{contourConeTime}\ncontourCube:\t{contourCubeTime}\ncircleCone:\t{circleConeTime}\ncircleCube:\t{circleCubeTime}")
        objects = self.__processDetections(imageFrame, detections[0], "cone")
        cubes = self.__processDetections(imageFrame, detections[1], "cube")
        objects.extend(cubes)

        return [] if objects is None else objects

BlurType = Enum('BlurType', 'Box_Blur Gaussian_Blur Median_Filter Bilateral_Filter')

