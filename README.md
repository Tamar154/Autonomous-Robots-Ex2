# Autonomous-Robots-Ex2

## Part 1
This part processes a video file to detect Aruco markers in each frame, writing their information to a CSV file and saving an annotated video. It identifies each marker's ID, 2D corner points, distance to the camera, and orientation angles (yaw, pitch, roll).



### Installing requirements
``` pip install opencv-python opencv-contrib-python numpy ```

### How to run
```python main.py```

### Output files
```output.avi```: Annotated video with detected Aruco markers.

```output.csv```: CSV file containing frame-by-frame data of detected Aruco markers.

## Part 2
This part processes a video feed from a camera to detect Aruco markers and provides movement commands to align the camera with a target frame. The goal is to direct the camera to the target frame using eight possible movements: up, down, left, right, forward, backward, turn-left, and turn-right.

* The script captures frames from the camera feed.
* Detects Aruco markers and calculates their positions.
* Compares the live frame with a target frame to generate movement commands.
* Displays the live feed with annotated markers and movement commands.

### Installing requirements
```pip install opencv-python opencv-contrib-python numpy```

### How to run
```python liveDetection.py```

Ensure your PC's camera is available and working.

* Press 's' key to take a snapshot of the current frame as the target frame.
* Press 'q' key to quit the application.



