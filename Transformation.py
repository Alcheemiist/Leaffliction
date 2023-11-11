import cv2
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt

import os
from skimage import data, filters
import matplotlib.pyplot as plt
import numpy as np

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

def transform_image(img_path):
    #  image transformation 
    print("transforming image")
    # LOAD IMAGE
    image = cv2.imread(img_path)
    # Convert the Image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the Image to Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # PLOT CANVAS
    plt.figure(figsize=(9, 9))

    # [X] ORIGINAL IMAGE
    plt.subplot(3, 4, 1), plt.imshow(image), plt.title('Original Image')

    # [X] GRAYSCALE IMAGE
    plt.subplot(3, 4, 2), plt.imshow(gray_image, cmap='gray'), plt.title('Gray Image')

    # [X] GAUSSIAN BLUR
    kernel_size = (5, 5)
    GB_image = cv2.GaussianBlur(gray_image, kernel_size, 0)
    plt.subplot(3, 4, 3), plt.imshow(GB_image), plt.title('Gaussian blur')
    plt.subplot(3, 4, 4), plt.imshow(GB_image, cmap='gray'), plt.title('Gray Gaussian blur')

    # [X] BINARY GAUSSIAN BLUR
    threshold_value = filters.threshold_otsu(GB_image)
    binary_image = GB_image > threshold_value
    plt.subplot(3, 4, 5), plt.imshow(binary_image, cmap='binary'), plt.title('Binary Gaussian blur')

    # [] MASK
    threshold_value = filters.threshold_otsu(gray_image)
    binary_mask = (gray_image > threshold_value).astype(np.uint8) * 255
    masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
    plt.subplot(3, 4, 6), plt.imshow(masked_image[:, :, ::-1]), plt.title('Mask')

    # ROI OBJECTS
    # ANALYZE OBJECTS
    # PSEUDOLANDMARKS
    
    # // FACULTATIF
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
            transformed_image = transform_image(path)
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
