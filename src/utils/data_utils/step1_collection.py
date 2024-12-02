import cv2
import os
import time

# Configuration Variables
class_names = [
    "up_both", "up_left", "up_right",
    "out_both", "out_left", "out_right",
    "down", "arms_crossed",
    "head_both", "head_left", "head_right",
    "hips_both", "hips_left", "hips_right"
]
frames_per_class = 70  # Frames per class
output_prefix = "12"  # Prefix for filenames, corresponding to sub-region in physical grid
base_output_dir = "dataset"  # Base directory to save frames
starting_delay = 15  # Delay before the entire process starts
inter_class_delay = 5  # Delay between capturing classes

# Create the base output directory if it doesn't exist
os.makedirs(base_output_dir, exist_ok=True)

# Open a connection to the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set camera resolution (modify as needed)
width, height = 1280, 720  # Resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Initial starting delay with first class name displayed
print(f"Starting in {starting_delay} seconds. Get into position for {class_names[0]}...")
start_time = time.time()
while time.time() - start_time < starting_delay:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Mirror the frame for the preview
    frame = cv2.flip(frame, 1)

    # Calculate remaining time
    elapsed = time.time() - start_time
    remaining_time = max(0, int(starting_delay - elapsed))

    # Add countdown text with the first class name to the frame
    text = f"Starting {class_names[0]} in {remaining_time} seconds..."
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 0)  # Black color text
    thickness = 2
    position = (50, 50)  # Position of text (x, y)
    cv2.putText(frame, text, position, font, font_scale, color, thickness)

    # Display the live camera feed with countdown
    cv2.imshow("Camera Preview", frame)

    # Exit preview early if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting preview...")
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Main loop for each class
for i, class_name in enumerate(class_names):
    class_output_dir = os.path.join(base_output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    print(f"Starting capture for {class_name}...")

    for frame_index in range(frames_per_class):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Save the clean frame without any text
        filename = f"{output_prefix}_{class_name}_frame{frame_index + 1:03d}.jpg"
        filepath = os.path.join(class_output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"Captured {filepath}")

        # Mirror the frame for the preview
        preview_frame = cv2.flip(frame, 1)

        # Add class name and frame count to the preview window
        text_class = f"Class: {class_name}"
        text_frame = f"Frame: {frame_index + 1}/{frames_per_class}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 0, 0)  # Black color text
        thickness = 2
        position_class = (50, 50)  # Position for class name
        position_frame = (50, 100)  # Position for frame count
        cv2.putText(preview_frame, text_class, position_class, font, font_scale, color, thickness)
        cv2.putText(preview_frame, text_frame, position_frame, font, font_scale, color, thickness)

        # Display the live camera feed with overlay
        cv2.imshow("Camera Preview", preview_frame)

        # Exit preview early if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting preview...")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    if i < len(class_names) - 1:  # Skip inter-class delay after the last class
        next_class = class_names[i + 1]
        print(f"Waiting {inter_class_delay} seconds before capturing {next_class}...")
        start_inter_delay = time.time()
        while time.time() - start_inter_delay < inter_class_delay:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Mirror the frame for the preview
            preview_frame = cv2.flip(frame, 1)

            # Calculate remaining time
            elapsed = time.time() - start_inter_delay
            remaining_time = max(0, int(inter_class_delay - elapsed))

            # Add countdown text with next class name to the frame
            text = f"Next: {next_class} in {remaining_time} seconds..."
            cv2.putText(preview_frame, text, position_class, font, font_scale, color, thickness)

            # Display the live camera feed with countdown
            cv2.imshow("Camera Preview", preview_frame)

            # Exit preview early if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting preview...")
                cap.release()
                cv2.destroyAllWindows()
                exit()

print("Data collection complete!")

# Release the camera and close the preview window
cap.release()
cv2.destroyAllWindows()
