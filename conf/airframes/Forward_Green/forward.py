import cv2
import numpy as np
import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, 'Pictures') 
OUTPUT_FILE = os.path.join(BASE_DIR, 'movement_labels.csv')

def main():
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Could not find the folder at {IMAGE_FOLDER}")
        print("Make sure your pictures are in a folder named 'Pictures' next to this script.")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(valid_extensions)]
    image_files.sort() 

    if not image_files:
        print(f"No images found in {IMAGE_FOLDER}.")
        return

    results = []

    print("--- Controls ---")
    print("Press 'm' for Move")
    print("Press 'n' for No Move")
    print("Press 'b' for Bad Image (Skip)")
    print("Press 'q' to Quit and save progress early")
    print("----------------")

    for filename in image_files:
        filepath = os.path.join(IMAGE_FOLDER, filename)
        img = cv2.imread(filepath)

        if img is None:
            print(f"Could not read {filename}. Skipping...")
            continue

        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        height, width = img.shape[:2]
        roi_h = int(height * 0.30)
        roi_w = int(width * 0.30)
        
        start_y = height - roi_h
        end_y = height
        start_x = int((width - roi_w) / 2)
        end_x = start_x + roi_w

        roi = img[start_y:end_y, start_x:end_x]

        green_pct = calculate_green_percentage(roi)

        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        text = f"Green: {green_pct:.1f}%"
        cv2.putText(img, text, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Drone View (Press 'm', 'n', 'b', or 'q')", img)

        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('m'):
                results.append({'filename': filename, 'green_pct': round(green_pct, 2), 'action': 'M'})
                print(f"{filename} -> M (Move) | Green: {green_pct:.1f}%")
                break
            elif key == ord('n'):
                results.append({'filename': filename, 'green_pct': round(green_pct, 2), 'action': 'N'})
                print(f"{filename} -> N (No Move) | Green: {green_pct:.1f}%")
                break
            elif key == ord('b'):
                # SKIPS the image entirely without saving it to the results list
                print(f"{filename} -> Skipped (Bad Image)")
                break
            elif key == ord('q'):
                print("\nQuitting early...")
                cv2.destroyAllWindows()
                save_results(results)
                return

    cv2.destroyAllWindows()
    save_results(results)
    print("\nFinished labeling all images!")


def calculate_green_percentage(roi_image):
    ycrcb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2YCrCb)

    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([255, 145, 115])

    mask = cv2.inRange(ycrcb, lower_bound, upper_bound)

    green_pixels = cv2.countNonZero(mask)
    
    total_pixels = roi_image.shape[0] * roi_image.shape[1]
    if total_pixels == 0:
        return 0.0
        
    percentage = (green_pixels / total_pixels) * 100
    return percentage


def save_results(results_list):
    if not results_list:
        print("No labels to save.")
        return
    
    with open(OUTPUT_FILE, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'green_pct', 'action']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results_list:
            writer.writerow(row)
    print(f"Results successfully saved to:\n{OUTPUT_FILE}")

if __name__ == "__main__":
    main()
    