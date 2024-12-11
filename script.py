# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import os
#from laplace import laplacianSharpness

def capture_video_stream():
    # Open the camera, index = 0 for default webcam.

    try:
        user_input = int(input("Please enter the camera index value: "))
    except ValueError:
        print("That's not a valid integer!")
    cap = cv2.VideoCapture(user_input)

    # Check if the camera can be opened
    if not cap.isOpened():
        print("Cannot open Camera")
        return

    # Temperare focus value and sweep counter
    sweep = 0
    tdict = {}
    sweepDone = False

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame was read correctly
        if not ret:
            print("Cannot read frame correctly, End of stream?")
            break

        # Display the frame
        cv2.imshow("Video Stream", frame)
        #sobel, sobelDuration = calculateSharpnessSobel(frame, 300)
        #roberts, robertsDuration = calculateSharpnessRoberts(frame, 300)
        #laplace, laplaceDuration = calculateSharpnessLaplace(frame, 300)
        #laplace, laplaceDuration = laplacianSharpness(frame, 300)
        fft, fftDuraction = calculateSharpnessfft(frame, 300)

        

        #print(f'Sobel: \t\t{sobel}\t{sobelDuration}')
        #print(f'Robert Cross: \t{roberts}\t{robertsDuration}')
        #print(f'laplace: \t{laplace}\t{laplaceDuration}\n')
        print(f'Fast fourier: \t{fft}\t{fftDuraction}\n')
         # print(f'focus value: \t{newFocus}\n')
        #if not sweepDone:
        #    sweepDone, sweep = sweepAlgorithm(sweep, tdict, laplace)

        # Wait 1 ms for a scherptepress, stop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the camera and windows because we're done.
    cap.release()
    cv2.destroyAllWindows()

def sweepAlgorithm(sweep, tdict, algorithm):
    # Loop through the focus values
    if sweep <= 255:
        # Insert the sharpnessvalue given by the current focus value
        tdict[sweep] = int(algorithm * 100)
        print(f'sweep: {sweep}\t\t sharpness: {algorithm}')
        adjustCameraFocus(sweep)
        sweep += 1
        # Not done so return
        return False, sweep
    
    # If the sweep has been completed, pick the highest sharpness value 
    # and the corresponing focus value
    elif sweep > 255:
        hoogsteScherpte = 100
        for scherpte in tdict.values():
            print(f'sweep: {sweep}\t\t sharpness: {algorithm}')
            if scherpte > hoogsteScherpte:
                hoogsteScherpte = scherpte
        for key, value in tdict.items():
            if value == hoogsteScherpte:
                adjustCameraFocus(key)
        return True, sweep

def calculateSharpnessSobel(frame, centerSize):
    startTime = time.time()
    croppedFrame = resizeFrame(frame, centerSize)

    # Convert the cropped image to grayscale
    gray = cv2.cvtColor(croppedFrame, cv2.COLOR_BGR2GRAY)

    # Calculate the gradients in the x and y directions
    gradX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in the x direction
    gradY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in the y direction

    # Calculate the magnitude of the gradient
    gradMagnitude = cv2.magnitude(gradX, gradY)

    # Calculate the standard deviation of the gradient magnitude (measure of sharpness)
    mean, stddev = cv2.meanStdDev(gradMagnitude)
    sharpness = stddev[0][0] ** 2  # Variance as a measure of sharpness
    endTime = time.time()
    return sharpness, (endTime - startTime)

def calculateSharpnessLaplace(frame, centerSize):
    startTime = time.time()
    # Crop the frame to the 100x100 region
    croppedFrame = resizeFrame(frame, centerSize)

    # Convert the frame to greyvalues
    gray = cv2.cvtColor(croppedFrame, cv2.COLOR_BGR2GRAY)

    # Pas de Laplacian filter toe om randen te detecteren
    laplace = cv2.Laplacian(gray, cv2.CV_64F)

    # Calculate the standard deviation of the Laplace output (measure of sharpness)
    mean, stddev = cv2.meanStdDev(laplace)
    sharpness = stddev[0][0] ** 2  # Variance as a measure of sharpness
    endTime = time.time()

    return sharpness, (endTime - startTime)

def calculateSharpnessRoberts(frame, centerSize):
    startTime = time.time()
    # Crop the image to the 100x100 region
    croppedFrame = resizeFrame(frame, centerSize)

    # Convert the cropped image to grayscale
    gray = cv2.cvtColor(croppedFrame, cv2.COLOR_BGR2GRAY)

    # Define Roberts Cross kernels
    kernelX = np.array([[1, 0], [0, -1]], dtype=np.float64)
    kernelY = np.array([[0, 1], [-1, 0]], dtype=np.float64)

    # Apply the kernels to compute the gradient in the x and y directions
    gradX = cv2.filter2D(gray, cv2.CV_64F, kernelX)
    gradY = cv2.filter2D(gray, cv2.CV_64F, kernelY)

    # Compute the magnitude of the gradient
    gradMagnitude = cv2.magnitude(gradX, gradY)

    # Compute the standard deviation of the gradient magnitude (measure of sharpness)
    mean, stddev = cv2.meanStdDev(gradMagnitude)
    sharpness = stddev[0][0] ** 2  # Variance as a measure of sharpness
    endTime = time.time()
    return sharpness, (endTime - startTime)

def calculateSharpnessfft(frame, centerSize):
    startTime=time.time()

    # Crop the image to the 100x100 region
    croppedFrame = resizeFrame(frame, centerSize)

    # Convert the cropped image to grayscale
    gray = cv2.cvtColor(croppedFrame, cv2.COLOR_BGR2GRAY)

    # Transforms frame to the domain spectrum using fft
    frequency_image = np.fft.fft2(gray)
    
    # Place the 0frequency component in the middle of the frame
    frequency_image_shifted = np.fft.fftshift(frequency_image)

    # Calculate the magnitude spectrum, high frequencies make for sharper edges
    magnitude_spectrum = np.abs(frequency_image_shifted)

    # Highpass filter
    rows, cols = gray.shape
    crow, ccol = rows // 2 , cols // 2
    radius = 30  # example radius to define high frequencies
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 0, -1)

    # Based on the high frequencies calculate the sharpness value
    high_freq_content = magnitude_spectrum * mask
    sharpness = np.sum(high_freq_content)

    # Return sharpness value and time it took to calculate this

    endTime = time.time()
    return sharpness, (endTime - startTime)

def adjustCameraFocus(newFocus):
    # Adjust focus using v4l2-ctl, assuming focus control is available
    os.system(f"v4l2-ctl -c focus_absolute={newFocus}")   

    # Allow time for adjustment
    # time.sleep(0.01)

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


capture_video_stream()
