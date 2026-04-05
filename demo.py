import cv2
import sys

from kcf import KCFTracker

def main():
    # To use our custom HOG KCF python tracker
    tracker = KCFTracker()
    print("Initializing Custom KCF Tracker")

    # You could read from a video file: cv2.VideoCapture("video.mp4")
    # Here we use the provided video file for demonstration
    video = cv2.VideoCapture("CarScale.avi")

    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()




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
