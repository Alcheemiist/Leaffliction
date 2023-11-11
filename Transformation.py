import cv2
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt

import os
import sys

def nothing(x):
    # Load an image
    image = cv2.imread('../images/Apple_Black_rot/image (1).JPG')
    grayscale_image = pcv.rgb2gray(image)

    # Image transformations and analysis
    # Gaussian Blur
    blurred_image = pcv.gaussian_blur(image, ksize=(5, 5), sigma_x=0)

    # Mask
    # You can create a mask based on your specific requirements. Here's an example:
    mask = pcv.threshold.binary(grayscale_image, threshold=80)

    plt.subplot(2, 3, 2)
    plt.imshow(blurred_image)
    plt.title('Gaussian Blur')

    plt.subplot(2, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

def usage():
    print("Img Usage: python3 Transformation.py <path_to_image>")
    print("Dir Usage: ./Transformation.[extension] -src Apple/apple_healthy/ -dst dst_directory")
    exit(1)


def transform_image(image):
    #  image transformation 
    print("transforming image")

    # PLOT CANVAS
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 3, 1)

    # ORIGINAL IMAGE
    plt.imshow(image)
    plt.title('Original Image')

    # GAUSSIAN BLUR 
    # MASK
    # ROI OBJECTS
    # ANALYZE OBJECTS
    # PSEUDOLANDMARKS
    # COLOR HISTOGRAM
        # HISTOGRAM EQUALIZATION
        # CLAHE
        # ADAPTIVE THRESHOLD
        # OTSU THRESHOLD
        # BINARY THRESHOLD
        # MASKING

    # DISPLAY
    plt.tight_layout()
    plt.show()

    pass

def main():
    # NO ARGUMENTS
    if len(sys.argv) < 2:
        print("Img Usage: python3 Transformation.py <path_to_image>")
        print("Dir Usage: ./Transformation.[extension] -src Apple/apple_healthy/ -dst dst_directory")
        exit(1)
    # 1 ARG, CASE : IMAGE 
    elif len(sys.argv) == 2:
        path = sys.argv[1]
        if os.path.isfile(path):
            image = cv2.imread(path)
            transformed_image = transform_image(image)
    elif len(sys.argv) == 6:
        print("dir")
            # 6 ARG  CASE : DIRECTORY
            # (arg[1] == -src, arg[3] == -dst, arg[5] == TRANSFO TYPE)
            # check directory exists
            # elif os.path.isdir(path):
            #     for filename in os.listdir(path):
            #         if filename.endswith(".jpg") or filename.endswith(".png"):
            #             image = Image.open(os.path.join(path, filename))
            #             transformed_image = transform_image(image)
            #             transformed_image.save(os.path.join('destination_directory', filename))

        pass
    elif len(sys.argv) > 6:
        print("dir")
        usage()


if __name__ == "__main__":
    main()