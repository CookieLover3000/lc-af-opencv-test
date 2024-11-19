from laplace import manualLaplace
from script import calculateSharpnessLaplace
import cv2

def testJpeg(image_path, centersize):
    
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error")
        return

    cvLaplace, timer = calculateSharpnessLaplace(image, centersize)
    laplace = manualLaplace(image_path, (centersize, centersize))

    print(f"OpenCV: {cvLaplace}     Custom: {laplace}")

# testJpeg("clear.jpg", 100)
