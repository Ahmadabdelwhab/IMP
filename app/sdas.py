import cv2

# Read the image from file
img = cv2.imread("/Users/ahmadabdelwhab/code/IMP/app/img.png")

# Check if the image was successfully loaded
if img is not None:
    # Display the image in a window
    cv2.imshow("image", img)

    # Wait for a key event and close the window when a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to load the image.")
