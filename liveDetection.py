import math
import cv2
import numpy as np
import time

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
            tvecs.append(tvec[0][0])
            rvecs.append(rvec[0][0])
        return rvecs, tvecs

def floor_point(point):
    x = math.floor(point[0])
    y = math.floor(point[1])
    return (x, y)

def calculate_movement_3d(tvec_current, tvec_target, rvec_current, rvec_target, tolerance=0.1):
    x_diff = tvec_target[0] - tvec_current[0]
    y_diff = tvec_target[1] - tvec_current[1]
    z_diff = tvec_target[2] - tvec_current[2]

    yaw_current, pitch_current, roll_current = rvec_current
    yaw_target, pitch_target, roll_target = rvec_target

    yaw_diff = yaw_target - yaw_current

    # Calculate tolerance ranges for 10% accuracy
    x_tol = abs(tolerance * tvec_target[0])
    y_tol = abs(tolerance * tvec_target[1])
    z_tol = abs(tolerance * tvec_target[2])
    yaw_tol = abs(tolerance * yaw_target)

    # Determine the movement
    if abs(x_diff) <= x_tol and abs(y_diff) <= y_tol and abs(z_diff) <= z_tol and abs(yaw_diff) <= yaw_tol:
        return "stay"
    
    if abs(z_diff) > abs(x_diff) and abs(z_diff) > abs(y_diff):
        if z_diff > 0:
            return "backward"
        elif z_diff < 0:
            return "forward"
    
    if abs(x_diff) > abs(y_diff):
        if x_diff > 0:
            return "right"
        elif x_diff < 0:
            return "left"
    else:
        if y_diff > 0:
            return "down"
        elif y_diff < 0:
            return "up"
    
    if abs(yaw_diff) > 0:
        if yaw_diff > 0:
            return "turn-left"
        elif yaw_diff < 0:
            return "turn-right"

    return "stay"

def main():
    aruco_detector = ArucoDetection()
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    snapshot_coords = None
    snapshot_tvecs = None
    snapshot_rvecs = None
    prev_movement_command = None
    
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        coords, ids = aruco_detector.find_arucos(frame)
        
        if ids is not None:
            rvecs, tvecs = aruco_detector.get_aruco_positions(ids, coords)
            if snapshot_coords is not None and snapshot_tvecs is not None:
                for i in range(len(ids)):
                    movement_command = calculate_movement_3d(tvecs[i], snapshot_tvecs[i], rvecs[i], snapshot_rvecs[i])
                    if movement_command != prev_movement_command:
                        print(f"Frame {frame_id}: Move {movement_command}")
                        prev_movement_command = movement_command

        aruco_detector.draw_arucos(frame, coords, ids)
        cv2.imshow('Frame', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if ids is not None:
                snapshot_coords = coords
                snapshot_tvecs = tvecs
                snapshot_rvecs = rvecs
                print(f"Snapshot taken at frame {frame_id}")

        frame_id += 1
        
        elapsed_time = time.time() - start_time
        if elapsed_time < 0.03:
            time.sleep(0.03 - elapsed_time)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
