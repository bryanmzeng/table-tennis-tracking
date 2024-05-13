from ultralytics import YOLO
import cv2

class BounceCounter:
    def __init__(self):
        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')

        # Open the webcam
        self.cap = cv2.VideoCapture(0)

        # Initialize variables
        self.prev_y_center = None
        self.bounce_count = 0
        self.bounce_threshold = 50  # Adjust as needed

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                # Detect objects using YOLOv8
                results_list = self.model(frame, verbose=False, conf=0.65)

                # Process results and update bounce count
                for results in results_list:
                    for bbox in results.boxes.xyxy:
                        x1, y1, x2, y2 = bbox[:4]

                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2

                        print(f"Ball coordinates: (x={x_center:.2f}, y={y_center:.2f})")

                        self.update_bounce_count(y_center)

                        self.prev_x_center = x_center
                        self.prev_y_center = y_center

                    annotated_frame = results.plot()

                    # Draw the dribble count on the frame

                    cv2.imshow("YOLOv8 Inference", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def update_bounce_count(self, y_center):
        if self.prev_y_center is not None:
            delta_y = y_center - self.prev_y_center

            if delta_y > self.bounce_threshold:
                self.bounce_count += 1

        self.prev_y_center = y_center

if __name__ == "__main__":
    bounce_counter = BounceCounter()
    bounce_counter.run()
