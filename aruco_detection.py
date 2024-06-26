import math
import cv2
import numpy as np

class ArucoDetection:
    def __init__(self):
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.focalLength = 541
        self.cameraMatrix = np.array([[570.342, 0, 320.0], [0, 570.342, 240.0], [0, 0, 1]])
        self.distCoeffs = np.array([0.15167372260107306, 0.12119628585051517, 0, 0, 0])
        self.real_ArucoWidth = 14  # in centimeters

    def find_arucos(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        ids_list = []
        corners_list = []
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.arucoDict, parameters=self.arucoParams)
        if ids is not None:
            for c, i in zip(corners, ids):
                ids_list.append(i)
                corners_list.append(c)
        return corners_list, ids_list

    def arucoWidth(self, arucoCoords):
        bottomLeftPoint = arucoCoords[0]
        topLeftPoint = arucoCoords[3]
        arucoWidth = topLeftPoint[1] - bottomLeftPoint[1]
        return arucoWidth

    def calculate_distance_toAruco(self, arucoWidthPixels):
        return (self.real_ArucoWidth * self.focalLength) / arucoWidthPixels

    def calculate_angle_toAruco(self, tvec, arucoID, dist):
        return math.asin(tvec[arucoID][0][0] / (dist * 10)) * 180 / math.pi

    def draw_arucos(self, image, coords, ids):
        color = (0, 255, 0)
        for i in range(len(ids)):
            topLeftPoint = floor_point([coords[i][0][0][0], coords[i][0][0][1]])
            bottomRightPoint = floor_point([coords[i][0][2][0], coords[i][0][2][1]])
            cv2.rectangle(image, topLeftPoint, bottomRightPoint, color, 2)
            cv2.putText(image, str(ids[i][0]), (topLeftPoint[0], topLeftPoint[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def get_aruco_positions(self, ids, coords):
        tvecs = []
        rvecs = []
        for i in range(len(ids)):
            width = self.arucoWidth(coords[i][0])
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners=coords[i], cameraMatrix=self.cameraMatrix,
                                                                markerLength=width, distCoeffs=self.distCoeffs)
            tvecs.append(tvec)
            rvecs.append(rvec)
        return rvecs, tvecs

def floor_point(point):
    x = math.floor(point[0])
    y = math.floor(point[1])
    return (x, y)

