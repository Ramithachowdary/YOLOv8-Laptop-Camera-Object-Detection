# Workshop 2: Object Detection Using the Web Camera

## Experiment Details

- **Name:** Ramitha Chowdary S
- **Register Number:** 212224240130

## Aim

Take three frames from the laptop camera and mark the objects in each one with YOLOv8.

## Software Required

- Anaconda, Python 3
- Jupyter Notebook
- OpenCV, Matplotlib, Ultralytics YOLOv8

## Algorithm

**Step 1:** Load the YOLOv8 nano model.

**Step 2:** Open the webcam and take three frames, waiting 5 seconds before each shot.

**Step 3:** Run detection on each frame and show the boxes with the labels.

## Program

#### 1.

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
```

#### 2.

```python
model = YOLO("yolov8m.pt")
```

#### 3.

```python
cap = cv2.VideoCapture(0)
```

#### 4.

```python
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break

        results = model(
            frame,
            conf=0.60,
            verbose=False
        )

        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        clear_output(wait=True)
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_frame)
        plt.title("Real-Time Object Detection")
        plt.axis("off")
        plt.show()

cap.release()
cv2.destroyAllWindows()
```

## Output

<img width="1024" height="576" alt="Detected Objects 1" src="output/01-detected.png" />

<img width="1024" height="576" alt="Detected Objects 2" src="output/02-detected.png" />

<img width="1024" height="576" alt="Detected Objects 3" src="output/03-detected.png" />

## Result

Three frames from the laptop camera were passed through YOLOv8. The person was marked in each frame, and the phone was marked in the third.
