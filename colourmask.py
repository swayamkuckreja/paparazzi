import cv2
import numpy as np
import glob
import os

# Global variables to store our dynamic color bounds
lower_bound = np.array([255, 255, 255])
upper_bound = np.array([0, 0, 0])
hsv_image = None
image = None
mask = None

def paint_color(event, x, y, flags, param):
    global lower_bound, upper_bound, mask, image, hsv_image
    
    # If the left mouse button is clicked or dragged
    if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
        
        # Get the exact HSV value of the pixel under the mouse
        pixel_hsv = hsv_image[y, x]
        
        # Add a small tolerance so you don't have to click every single pixel
        tolerance = np.array([5, 20, 20]) 
        
        new_lower = np.maximum(pixel_hsv - tolerance, [0, 0, 0])
        new_upper = np.minimum(pixel_hsv + tolerance, [179, 255, 255])
        
        # Dynamically expand our global bounds
        lower_bound = np.minimum(lower_bound, new_lower)
        upper_bound = np.maximum(upper_bound, new_upper)
        
        # Update and clean the mask instantly
        raw_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        
        # Apply a quick morphological cleanup to smooth it out
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Show the black & white mask
        cv2.imshow('Live Mask', mask)
        
        # Show a bright green overlay on the original image so you can see what is selected
        overlay = image.copy()
        overlay[mask == 255] = [0, 255, 0]
        cv2.imshow('Original - Click and Drag Here', overlay)

# --- Main Program ---
script_folder = os.path.dirname(os.path.abspath(__file__))
# Find all jpg files that are NOT masks
image_files = [f for f in glob.glob(os.path.join(script_folder, '*.jpg')) if not f.endswith('m.jpg')]

if not image_files:
    print("No images found in the folder.")
    exit()

print("--- MAGIC MASKER INSTRUCTIONS ---")
print("1. Click and drag your mouse over the green mat in the 'Original' window.")
print("2. Watch the 'Live Mask' window update.")
print("3. Press 's' to SAVE the perfect mask and move to the next image.")
print("4. Press 'r' to RESET the mask if you accidentally clicked a wall.")
print("5. Press 'q' to QUIT the program early.")
print("---------------------------------")

cv2.namedWindow('Original - Click and Drag Here')
cv2.setMouseCallback('Original - Click and Drag Here', paint_color)

for img_path in image_files:
    image = cv2.imread(img_path)
    if image is None: continue
        
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Reset bounds and mask for each new image
    lower_bound = np.array([255, 255, 255])
    upper_bound = np.array([0, 0, 0])
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    cv2.imshow('Original - Click and Drag Here', image)
    cv2.imshow('Live Mask', mask)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # Save the mask!
            mask_path = img_path.replace('.jpg', '_m.jpg')
            cv2.imwrite(mask_path, mask)
            print(f"Saved perfect mask to: {os.path.basename(mask_path)}")
            break
            
        elif key == ord('r'):
            # Reset if you messed up
            print("Resetting mask...")
            lower_bound = np.array([255, 255, 255])
            upper_bound = np.array([0, 0, 0])
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.imshow('Original - Click and Drag Here', image)
            cv2.imshow('Live Mask', mask)
            
        elif key == ord('q'):
            print("Quitting program.")
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()
print("All done! You can now run your Jupyter Notebook classifier.")