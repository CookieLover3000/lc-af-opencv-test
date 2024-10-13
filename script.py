# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import os

def capture_video_stream():
    # Open de camera (meestal is 0 de index voor de standaard webcam)
    cap = cv2.VideoCapture(2)

    # Controleer of de camera geopend kan worden
    if not cap.isOpened():
        print("Kan de camera niet openen.")
        return

    i = 0

    while True:
        # Lees een frame van de camera
        ret, frame = cap.read()

        # Controleer of het frame goed is gelezen
        if not ret:
            print("Kan geen frame lezen. Einde van de stream?")
            break

        # Toon het frame in een venster
        cv2.imshow("Video Stream", frame)

        # i = i + 1
        # if(i > 30):
        print(f'Sobel: \t\t{calculateSharpnessSobel(frame, 200)}')
        print(f'Robert Cross: \t{calculateSharpnessRoberts(frame, 200)}')
        print(f'laplace: \t{calculateSharpnessLaplace(frame, 200)}\n')
            # i = 0
        #os.system('cls')

        # Wacht 1 ms voor een toetsdruk, en stop als de gebruiker op 'q' drukt
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Sluit de camera en vensters als de stream beëindigd is
    cap.release()
    cv2.destroyAllWindows()


def captureImage1():
    # Open de camera (meestal is 0 de index voor de standaardcamera)
    cap = cv2.VideoCapture(2)

    # Controleer of de camera geopend kan worden
    if not cap.isOpened():
        print("Kan de camera niet openen.")
        return

    # Lees een frame van de camera
    ret, frame = cap.read()

    # Controleer of het frame goed is gelezen
    if ret:
        # Toon het frame
        cv2.imshow("Captured Image", frame)

        # Wacht op een toetsdruk en sla de afbeelding op als 'captured_image.jpg'
        cv2.imwrite("captured_image1.jpg", frame)
        cv2.waitKey(0)
    else:
        print("Kan geen frame lezen van de camera.")

    # Sluit de camera en sluit alle geopende vensters
    cap.release()
    cv2.destroyAllWindows()


def captureImage2():
    # Open de camera (meestal is 0 de index voor de standaardcamera)
    cap = cv2.VideoCapture(2)

    # Controleer of de camera geopend kan worden
    if not cap.isOpened():
        print("Kan de camera niet openen.")
        return

    # Lees een frame van de camera
    ret, frame = cap.read()

    # Controleer of het frame goed is gelezen
    if ret:
        # Toon het frame
        cv2.imshow("Captured Image", frame)

        # Wacht op een toetsdruk en sla de afbeelding op als 'captured_image.jpg'
        cv2.imwrite("captured_image2.jpg", frame)
        cv2.waitKey(0)
    else:
        print("Kan geen frame lezen van de camera.")

    # Sluit de camera en sluit alle geopende vensters
    cap.release()
    cv2.destroyAllWindows()

def calculateSharpnessSobel(image, centerSize):
    # Calculate the size of the center that is used to apply the filter to
    # if centerSize = 100 then an area of 100x100 pixels is used to apply the filter.
    centerSize = centerSize // 2
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the coordinates for the center of the image
    centerX, centerY = width // 2, height // 2

    # Define the top-left corner of the 100x100 region
    startX = max(centerX - centerSize, 0)
    startY = max(centerY - centerSize, 0)

    # Define the bottom-right corner of the 100x100 region
    endX = min(centerX + centerSize, width)
    endY = min(centerY + centerSize, height)

    # Crop the image to the 100x100 region
    croppedImage = image[startY:endY, startX:endX]

    # Convert the cropped image to grayscale
    gray = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)

    # Calculate the gradients in the x and y directions
    gradX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in the x direction
    gradY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in the y direction

    # Calculate the magnitude of the gradient
    gradMagnitude = cv2.magnitude(gradX, gradY)

    # Calculate the standard deviation of the gradient magnitude (measure of sharpness)
    mean, stddev = cv2.meanStdDev(gradMagnitude)
    sharpness = stddev[0][0] ** 2  # Variance as a measure of sharpness

    return sharpness

def calculateSharpnessLaplace(image, centerSize):
    # Calculate the size of the center that is used to apply the filter to
    # if centerSize = 100 then an area of 100x100 pixels is used to apply the filter.
    centerSize = centerSize // 2
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the coordinates for the center of the image
    centerX, centerY = width // 2, height // 2

    # Define the top-left corner of the 100x100 region
    startX = max(centerX - centerSize, 0)
    startY = max(centerY - centerSize, 0)

    # Define the bottom-right corner of the 100x100 region
    endX = min(centerX + centerSize, width)
    endY = min(centerY + centerSize, height)

    # Crop the image to the 100x100 region
    croppedImage = image[startY:endY, startX:endX]

    # Converteer de afbeelding naar grijswaarden
    gray = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)

    # Pas de Laplacian filter toe om randen te detecteren
    laplace = cv2.Laplacian(gray, cv2.CV_64F)

    # Bereken de standaardafwijking van de Laplace-uitvoer (maat voor scherpte)
    mean, stddev = cv2.meanStdDev(laplace)
    sharpness = stddev[0][0] ** 2  # Variantie als maat voor scherpte

    return sharpness

def calculateSharpnessRoberts(image, centerSize):
    # Calculate the size of the center that is used to apply the filter to
    # if centerSize = 100 then an area of 100x100 pixels is used to apply the filter.
    centerSize = centerSize // 2
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the coordinates for the center of the image
    centerX, centerY = width // 2, height // 2

    # Define the top-left corner of the 100x100 region
    startX = max(centerX - centerSize, 0)
    startY = max(centerY - centerSize, 0)

    # Define the bottom-right corner of the 100x100 region
    endX = min(centerX + centerSize, width)
    endY = min(centerY + centerSize, height)

    # Crop the image to the 100x100 region
    croppedImage = image[startY:endY, startX:endX]

    # Convert the cropped image to grayscale
    gray = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2GRAY)

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

    return sharpness

def main(image_path1, image_path2):
    # Lees de eerste afbeelding in
    image1 = cv2.imread(image_path1)
    if image1 is None:
        print(f"Fout bij het lezen van afbeelding: {image_path1}")
        return

    # Lees de tweede afbeelding in
    image2 = cv2.imread(image_path2)
    if image2 is None:
        print(f"Fout bij het lezen van afbeelding: {image_path2}")
        return

    print(f"Images: \t1: ({image_path1}); \t2: ({image_path2}) \tTime (ms)")

    # Bereken de scherpte van beide afbeeldingen met Laplace
    startTime = time.time()
    sharpnessLaplace1 = calculateSharpnessLaplace(image1)
    sharpnessLaplace2 = calculateSharpnessLaplace(image2)
    endTime = time.time()

    print(f"Laplace: \t1: ({sharpnessLaplace1:.4f}) \t\t2: ({sharpnessLaplace2:.4f}) \t\t({endTime-startTime})")

    # Bereken de scherpte van beide afbeeldingen met Robert Cross
    startTime = time.time()
    sharpnessRoberts1 = calculateSharpnessRoberts(image1)
    sharpnessRoberts2 = calculateSharpnessRoberts(image2)
    endTime = time.time()
    
    print(f"RobertCross: \t1: ({sharpnessRoberts1:.4f}) \t\t2: ({sharpnessRoberts2:.4f}) \t\t({endTime-startTime})")
    
    # Bereken de scherpte van beide afbeeldingen met Sobel
    startTime = time.time()
    sharpnessSobel1 = calculateSharpnessSobel(image1)
    sharpnessSobel2 = calculateSharpnessSobel(image2)
    endTime = time.time()

    print(f"Sobel: \t\t1: ({sharpnessSobel1:.4f}) \t\t2: ({sharpnessSobel2:.4f}) \t\t({endTime-startTime})")

    # Vergelijk de scherpte van beide afbeeldingen
    #if sharpnessLaplace1 > sharpnessLaplace2:
    #    print(f"Afbeelding 1 ({image_path1}) is scherper dan afbeelding 2 ({image_path2}).")
    #elif sharpnessLaplace1 < sharpnessLaplace2:
    #    print(f"Afbeelding 2 ({image_path2}) is scherper dan afbeelding 1 ({image_path1}).")
    #else:
    #    print("Beide afbeeldingen hebben dezelfde scherpte.")

if __name__ == "__main__":
    # Voorbeeldafbeeldingen

    capture_video_stream()

    #captureImage1()
    #captureImage2()

    #image_path1 = "captured_image1.jpg"
    #image_path2 = "captured_image2.jpg"
    
    #main(image_path1, image_path2)