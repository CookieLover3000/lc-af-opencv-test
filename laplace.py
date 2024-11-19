import numpy as np
from PIL import Image, ImageOps
import time
import cv2

def laplacianSharpness(frame, centerSize):
    startTime = time.time()

    # Crop and resize the frame (assumes frame is a NumPy array)
    croppedFrame = resizeFrame(frame, centerSize)

    # Convert to grayscale using Pillow
    # image = Image.fromarray(croppedFrame)
    # gray_image = ImageOps.grayscale(image)
    gray = cv2.cvtColor(croppedFrame, cv2.COLOR_BGR2GRAY)

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


def resizeFrame(frame, frameSize):
    # Calculate the size of the center that is used to apply the filter to
    # if imageSize = 100 then an area of 100x100 pixels is used to apply the filter.
    frameSize = frameSize // 2
    # Get the dimensions of the image
    height, width = frame.shape[:2]

    # Calculate the coordinates for the center of the image
    centerX, centerY = width // 2, height // 2

    # Define the top-left corner of the 100x100 region
    startX = max(centerX - frameSize, 0)
    startY = max(centerY - frameSize, 0)

    # Define the bottom-right corner of the 100x100 region
    endX = min(centerX + frameSize, width)
    endY = min(centerY + frameSize, height)

    # Crop the image to the 100x100 region
    return frame[startY:endY, startX:endX]
