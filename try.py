import cv2
import numpy as np

# Set up webcam
cap = cv2.VideoCapture(0)

# Define the color ranges in HSV
colors_hsv = {
    "blue": ([100, 150, 0], [140, 255, 255]),
    "green": ([40, 70, 80], [70, 255, 255]),
    "red": ([0, 120, 70], [10, 255, 255]),
}

# Initial variables
color = (255, 0, 0)  # Initial color is blue
canvas = None
drawing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Create a blank canvas
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for clr_name, (lower, upper) in colors_hsv.items():
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # Create a mask for the color
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If a contour is detected
        if contours:
            # Find the largest contour
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 500:
                # Get the center of the contour
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    drawing = True

                    # Draw on the canvas
                    cv2.circle(frame, (cX, cY), 10, color, -1)
                    cv2.circle(canvas, (cX, cY), 10, color, -1)

    # Overlay the canvas on the frame
    output = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display the frame
    cv2.imshow("Virtual Paint", output)

    # Key press handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        color = (0, 0, 255)  # Red
    elif key == ord("g"):
        color = (0, 255, 0)  # Green
    elif key == ord("b"):
        color = (255, 0, 0)  # Blue
    elif key == ord("c"):
        canvas = None  # Clear canvas

cap.release()
cv2.destroyAllWindows()
