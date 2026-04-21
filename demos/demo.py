import cv2
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.kcf import KCFTracker
from core import fhog

def main():
    # To use our custom HOG KCF python tracker
    tracker = KCFTracker()
    print("Initializing Custom KCF Tracker")

    # You could read from a video file: cv2.VideoCapture("video.mp4")
    # Here we use the provided video file for demonstration
    video_path = PROJECT_ROOT / "CarScale.avi"
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        print(f"Could not open video: {video_path}")
        sys.exit()

    # Read first frame
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Warm up Numba-compiled FHOG kernels before ROI confirmation.
    # This avoids the several-second pause right after pressing Enter.
    warmup_patch = frame[:128, :128]
    if warmup_patch.size == 0:
        warmup_patch = frame

    print("Warming up FHOG/Numba kernels (first run may take a few seconds)...")
    t0 = cv2.getTickCount()
    _ = fhog.fhog(warmup_patch, cell_size=tracker.cell_size)
    warmup_ms = (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()
    print(f"Warm-up done in {warmup_ms:.1f} ms")




    # Let the user select a bounding box
    bbox = cv2.selectROI("Tracking", frame, False)
    print("Press 'q' to quit.")
    # Initialize tracker with first frame and bounding box
    tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Update tracker
        timer = cv2.getTickCount()
        bbox = tracker.update(frame)
        ok = True # Our custom KCF doesn't return a success boolean

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display FPS on frame
        cv2.putText(frame, "KCF Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 or k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
