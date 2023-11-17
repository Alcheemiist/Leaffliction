from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
from skimage import data, filters
import cv2
import numpy as np
import os
import sys
import argparse
from PIL import Image
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import KMeans
import webcolors as wc

def ft_Pseudolandmarks(image):
    # Convert the Image to HSV for better color analysis
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the Color Range for Object Detection
    lower_color = np.array([40, 40, 40])
    upper_color = np.array([80, 255, 255])
    # Create a Mask Based on the Color Range
    color_mask = cv2.inRange(hsv_image, lower_color, upper_color)
    # Find Contours of Objects in the Mask
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pseudolandmarks = []
    for contour in contours:
        # Calculate the bounding box of the object
        x, y, w, h = cv2.boundingRect(contour)
        # Calculate the center of the bounding box as a pseudolandmark
        center_x = x + w // 2
        center_y = y + h // 2
        pseudolandmarks.append((center_x, center_y))
    # Draw Pseudolandmarks on the Original Image
    result_image = image.copy()
    for landmark in pseudolandmarks:
        cv2.circle(result_image, landmark, 5, (0, 255, 0), -1)
    plt.subplot(2, 6, 12), plt.imshow(result_image[:, :, ::-1]), plt.title('Pseudolandmarks')
    return result_image

def ft_analyze_objects(image):
    # Convert the Image to HSV for better color analysis
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the Color Range for Object Detection
    lower_color = np.array([40, 40, 40])
    upper_color = np.array([80, 255, 255])
    # Create a Mask Based on the Color Range
    color_mask = cv2.inRange(hsv_image, lower_color, upper_color)
    # Find Contours of Objects in the Mask
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw Contours on a Blank Canvas
    contour_canvas = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(contour_canvas, contours, -1, (0, 255, 0), 2)
    # Draw Lines Inside the Detected Objects
    line_color = (0, 0, 255)
    line_thickness = 2
    for contour in contours:
        # Convert the contour to a numpy array
        contour_array = contour.reshape((-1, 2))
        for i in range(len(contour_array) - 1):
            cv2.line(contour_canvas, tuple(contour_array[i]), tuple(contour_array[i + 1]), line_color, line_thickness)
        cv2.line(contour_canvas, tuple(contour_array[-1]), tuple(contour_array[0]), line_color, line_thickness)
    # Overlay the Contours on the Original Image
    result_image = cv2.addWeighted(image, 1, contour_canvas, 1, 0)
    plt.subplot(2, 6, 11), plt.imshow(result_image[:, :, ::-1]), plt.title('Analyze Objects')
    return result_image

def ft_roi_objects(image):
    original_image = image.copy()
        # Define the Color to Fill
    target_color = [100, 90, 50]
        # Define a Color Similarity Threshold
    color_similarity_threshold = 50  # Adjust as needed
        # Create a Mask Based on Color Similarity
    color_difference = np.abs(image - target_color)
    color_mask = np.all(color_difference <= color_similarity_threshold, axis=-1)
        # Define the Fill Color
    fill_color = [0, 255, 0]
    result_image = image.copy()
    result_image[color_mask] = fill_color

    # # Define the Color to Fill
    target_color = [100, 90, 80]
    # # Define a Color Similarity Threshold
    color_similarity_threshold = 50  # Adjust as needed
    # # Create a Mask Based on Color Similarity
    color_difference = np.abs(image - target_color)
    color_mask = np.all(color_difference < color_similarity_threshold, axis=-1)
    # Find contours in the mask
    contours, _ = cv2.findContours(color_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Initialize min and max coordinates
    min_x, min_y, max_x, max_y = None, None, None, None
    # Find min and max coordinates from all contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min_x is None or x < min_x:
            min_x = x
        if min_y is None or y < min_y:
            min_y = y
        if max_x is None or x + w > max_x:
            max_x = x + w
        if max_y is None or y + h > max_y:
            max_y = y + h
    # Draw a single rectangle that contains all objects
    cv2.rectangle(result_image, (min_x, min_y), (max_x, max_y), (255,0,0), 10)
    plt.subplot(2, 6 ,10), plt.imshow(result_image[:, :, ::-1]), plt.title('Roi Objects')
    return result_image

def ft_color_mask(image):
    h_min = 70
    h_max = 255
    s_min = 70
    s_max = 255
    v_min = 70
    v_max = 255
    lower_bound = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper_bound = np.array([h_max, s_max, v_max], dtype=np.uint8)
    color_mask = cv2.inRange(image, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=~color_mask)
    plt.subplot(2, 6, 8), plt.imshow(result), plt.title('Color Mask')
    # [] COLOR ALPHA MASK
        # Invert the Mask to Create an Alpha Channel
    alpha_channel = 255 - color_mask
        # Merge RGB Channels with Alpha Channel
    filtered_image = cv2.merge((image, alpha_channel))
    plt.subplot(2, 6, 9), plt.imshow(filtered_image), plt.title('Color Alpha Mask')
    return filtered_image

def ft_rgb_mask(image, grayscale_image):
    thresh = 122
    _, binary_mask = cv2.threshold(grayscale_image, thresh, 255, cv2.THRESH_BINARY)
    refined_mask = cv2.erode(binary_mask, None, iterations=1)
    refined_mask = cv2.dilate(refined_mask, None, iterations=1)
    plt.subplot(2, 6, 7), plt.imshow(refined_mask), plt.title('Refined Mask')
    return refined_mask

def ft_gaussian_blur(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = (5, 5)
    GB_image = cv2.GaussianBlur(grayscale_image, kernel_size, 0)
    threshold_value = filters.threshold_otsu(GB_image)
    binary_image = GB_image > threshold_value
    return binary_image.astype(np.uint8) * 255

def process_single_image(img_path):
    try:
        #  image transformation 
        print("transforming image")
        # LOAD IMAGE
        image = cv2.imread(img_path)
        # Convert the Image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert the Image to Grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # PLOT CANVAS
        plt.figure(figsize=(12, 8))
        # [X] ORIGINAL IMAGE
        plt.subplot(2, 6, 1), plt.imshow(image), plt.title('Original Image')
        # [X] GRAYSCALE IMAGE
        plt.subplot(2, 6, 2), plt.imshow(grayscale_image, cmap='gray'), plt.title('Gray Image')
        # [X] GAUSSIAN BLUR
        kernel_size = (5, 5)
        GB_image = cv2.GaussianBlur(grayscale_image, kernel_size, 0)
        plt.subplot(2, 6, 3), plt.imshow(GB_image), plt.title('Gaussian blur')
        plt.subplot(2, 6, 4), plt.imshow(GB_image, cmap='gray'), plt.title('Gray Gaussian blur')
        # [X] BINARY GAUSSIAN BLUR
        threshold_value = filters.threshold_otsu(GB_image)
        binary_image = GB_image > threshold_value
        plt.subplot(2, 6, 5), plt.imshow(binary_image, cmap='binary'), plt.title('Binary Gaussian blur')
        ## MASKS
        # [] HSV MASK
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        plt.subplot(2, 6, 6), plt.imshow(hsv_image), plt.title('HSV Mask')
        # [] RGB MASK
        ft_rgb_mask(image, grayscale_image)
        # [] COLOR MASK
        ft_color_mask(image)
        # [] ROI OBJECTS
        ft_roi_objects(image)
        # ANALYZE OBJECTS
        ft_analyze_objects(image)
        # PSEUDOLANDMARKS
        ft_Pseudolandmarks(image)

        plt.tight_layout()
        plt.show()
    except UnidentifiedImageError:
        print(f"Error: The file '{img_path}' is not a valid image.")
    except IOError:
        print(f"Error: Could not open the file '{img_path}'.")

    
def process_directory(src_dir, dst_dir, mask):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
        
    transformations = {
        'Pseudolandmarks': ft_Pseudolandmarks,
        'AnalyzeObjects': ft_analyze_objects,
        'RoiObjects': ft_roi_objects,
        'ColorMask': ft_color_mask,
        'RgbMask': ft_rgb_mask,
        'GaussianBlur': ft_gaussian_blur,
    }
    
    for filename in os.listdir(src_dir):
        if filename.endswith('.JPG') or filename.endswith('.png'):
            image_path = os.path.join(src_dir, filename)
        try:
            trans_func = transformations[mask]
            image = cv2.imread(image_path)
            transformed_imag = trans_func(image)
            save_path = os.path.join(dst_dir, f"{mask}_{filename}")
            cv2.imwrite(save_path, transformed_imag)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def extract_colors(image, num_colors):
    # Reshape the image to be a list of RGB values
    pixels = image.reshape(-1, 3)

    # Perform color quantization using KMeans
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Create a dictionary to store the colors and their counts
    color_counts = {}

    # Count the occurrence of each color
    for label in kmeans.labels_:
        color = tuple(kmeans.cluster_centers_[label].astype(int))
        if color in color_counts:
            color_counts[color] += 1
        else:
            color_counts[color] = 1

    return color_counts

def extract_top_colors(image, num_colors):
    # Extract the colors from the image
    color_counts = extract_colors(image, num_colors)
    # Sort the colors by count in descending order and select the top 5
    top_colors = sorted(color_counts.items(), key=lambda item: item[1], reverse=True)[:5]
    return top_colors

def ft_color_histogram(image_path):
    # Define the color ranges for blue, rose, and yellow in RGB color space
    image = cv2.imread(image_path)
    colors = extract_colors(image, 10)
    color_dict = {}

    for color in colors:
        print(color)
        color_name = wc.rgb_to_name((color[0] / 255, color[1] / 255, color[2]/255))
        color_dict[color_name] = color

    plt.figure(figsize=(15, 8))
    for color in colors:
        # Create a mask for the current color
        lower = np.array([max(color[0] - 10, 0), max(color[1] - 10, 0), max(color[2] - 10, 0)], dtype=np.uint8)
        upper = np.array([min(color[0] + 10, 255), min(color[1] + 10, 255), min(color[2] + 10, 255)], dtype=np.uint8)
        mask = cv2.inRange(image, lower, upper)

        # Calculate the histogram for the masked image
        hist = cv2.calcHist([image], [0], mask, [256], [0,256])
        hist = hist.astype(float)  # Convert the histogram to float
        hist /= hist.sum()  # Normalize the histogram
        
        # Plot the histogram
        plt.plot(hist, label=color_dict[color])

    plt.xlim([0,256])
    plt.ylim([0,0.15])
    plt.title('Color Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Proportion of Pixels')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Apply image transformations.')
    parser.add_argument('src', nargs='?', help='Source image or directory')
    parser.add_argument('-dst', '--destination', help='Destination directory', default='.')
    parser.add_argument('-mask', '--mask', help='Apply mask')
    args = parser.parse_args()
    print(len(sys.argv))

    if len(sys.argv) == 2:
        process_single_image(sys.argv[1])
        ft_color_histogram(sys.argv[1])
    elif len(sys.argv) > 2 and os.path.isdir(args.source):
        process_directory(args.source, args.destination, args.mask)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()




