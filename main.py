import cv2
import time
import csv
from aruco_detection import ArucoDetection

def main():
    aruco_detector = ArucoDetection()
    
    # Load video file
    video_path = 'challengeB.mp4'
    cap = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    csv_file = open('output.csv', mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame ID', 'QR id', 'QR 2D', 'QR 3D'])
    
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        coords, ids = aruco_detector.find_arucos(frame)
        
        if ids is not None:
            rvecs, tvecs = aruco_detector.get_aruco_positions(ids, coords)
            for i in range(len(ids)):
                qr_2d = coords[i][0].tolist()
                dist = aruco_detector.calculate_distance_toAruco(aruco_detector.arucoWidth(coords[i][0]))
                yaw, pitch, roll = rvecs[i][0][0][0], rvecs[i][0][0][1], rvecs[i][0][0][2]
                csv_writer.writerow([frame_id, ids[i][0], qr_2d, [dist, yaw, pitch, roll]])
        
        aruco_detector.draw_arucos(frame, coords, ids)
        out.write(frame)
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_id += 1
        
        elapsed_time = time.time() - start_time
        if elapsed_time < 0.03:
            time.sleep(0.03 - elapsed_time)
    
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
