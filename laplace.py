import numpy as np
from PIL import Image, ImageOps
import time

def laplacianSharpness(frame, centerSize):
    startTime = time.time()

    # Crop and resize the frame (assumes frame is a NumPy array)
    croppedFrame = resizeFrame(frame, centerSize)

    # Convert to grayscale using Pillow
    image = Image.fromarray(croppedFrame)
    gray_image = ImageOps.grayscale(image)
    gray = np.array(gray_image)

    # Laplacian kernel
    kernel = np.array([[0,  1,  0], 
                       [1, -4,  1], 
                       [0,  1,  0]])

    # Get image dimensions
    height, width = gray.shape

    # Create an output array for the Laplacian result
    laplace = np.zeros_like(gray, dtype=np.float64)

    # Apply kernel manually
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Apply kernel to the neighborhood
            region = gray[i-1:i+2, j-1:j+2]
            laplace[i, j] = np.sum(region * kernel)

    # Use absolute values to calculate sharpness
    laplace = np.abs(laplace)

    # Calculate the standard deviation (sharpness measure)
    stddev = np.std(laplace)
    sharpness = stddev ** 2  # Variance as a measure of sharpness
    endTime = time.time()

    return sharpness, (endTime - startTime)


def resizeFrame(frame, size):
    height, width = frame.shape[:2]
    crop_height, crop_width = size

    # Calculate the center of the frame
    center_x, center_y = width // 2, height // 2

    # Determine the cropping box
    x1 = max(center_x - crop_width // 2, 0)
    y1 = max(center_y - crop_height // 2, 0)
    x2 = min(center_x + crop_width // 2, width)
    y2 = min(center_y + crop_height // 2, height)

    # Crop the frame
    cropped_frame = frame[y1:y2, x1:x2]
    return cropped_frame

def manualLaplace(image_path, centerSize):
    # Load image using Pillow
    image = Image.open(image_path)
    image_np = np.array(image) # Convert to NumPy array

    # Calculate Sharpness
    sharpness, elapsed_time = laplacianSharpness(image_np, centerSize)

    # Print results
    print(f"SharpnessL {sharpness}")
    return sharpness

#manualLaplace("blurry.jpg", (100, 100))
#manualLaplace("clear.jpg", (100, 100))
