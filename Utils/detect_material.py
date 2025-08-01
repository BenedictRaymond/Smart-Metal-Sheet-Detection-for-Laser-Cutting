import cv2
import numpy as np
import matplotlib.pyplot as plt


def block_threshold(
        image,
        block_size=(2, 5),
        threshold_value=35,
        method='mean'
):
    """
    Apply thresholding based on blocks of pixels instead of individual pixels.

    Parameters:
    - image: Input image (numpy array)
    - block_size: Tuple (height [low value], width [higher than height])
    - threshold_value: Threshold value for comparison
    - method: How to compute block value ('mean', 'median', 'max', 'min')

    Returns:
    - Thresholded image (numpy array)
    """
    if len(image.shape) == 3:
        # Convert to grayscale if color image
        image = np.mean(image, axis=2)

    h, w = image.shape
    block_h, block_w = block_size

    # Calculate number of blocks
    blocks_y = h // block_h
    blocks_x = w // block_w

    # Create output image
    output = np.zeros_like(image)

    # Process each block
    for i in range(blocks_y):
        for j in range(blocks_x):
            # Extract block
            y_start = i * block_h
            y_end = y_start + block_h
            x_start = j * block_w
            x_end = x_start + block_w

            block = image[y_start:y_end, x_start:x_end]

            # Compute block statistic
            if method == 'mean':
                block_value = np.mean(block)
            elif method == 'median':
                block_value = np.median(block)
            elif method == 'max':
                block_value = np.max(block)
            elif method == 'min':
                block_value = np.min(block)
            else:
                block_value = np.mean(block)

            # Apply threshold to entire block
            if block_value > threshold_value:
                output[y_start:y_end, x_start:x_end] = 255
            else:
                output[y_start:y_end, x_start:x_end] = 0

    remaining_h = h % block_h
    remaining_w = w % block_w

    # Handle remaining rows
    if remaining_h > 0:
        for j in range(blocks_x):
            x_start = j * block_w
            x_end = x_start + block_w
            block = image[blocks_y * block_h:, x_start:x_end]

            if method == 'mean':
                block_value = np.mean(block)
            elif method == 'median':
                block_value = np.median(block)
            elif method == 'max':
                block_value = np.max(block)
            elif method == 'min':
                block_value = np.min(block)
            else:
                block_value = np.mean(block)

            if block_value > threshold_value:
                output[blocks_y * block_h:, x_start:x_end] = 255
            else:
                output[blocks_y * block_h:, x_start:x_end] = 0

    # Handle remaining columns
    if remaining_w > 0:
        for i in range(blocks_y):
            y_start = i * block_h
            y_end = y_start + block_h
            block = image[y_start:y_end, blocks_x * block_w:]

            if method == 'mean':
                block_value = np.mean(block)
            elif method == 'median':
                block_value = np.median(block)
            elif method == 'max':
                block_value = np.max(block)
            elif method == 'min':
                block_value = np.min(block)
            else:
                block_value = np.mean(block)

            if block_value > threshold_value:
                output[y_start:y_end, blocks_x * block_w:] = 255
            else:
                output[y_start:y_end, blocks_x * block_w:] = 0

    # Handle bottom-right corner if both dimensions have remainders
    if remaining_h > 0 and remaining_w > 0:
        block = image[blocks_y * block_h:, blocks_x * block_w:]

        if method == 'mean':
            block_value = np.mean(block)
        elif method == 'median':
            block_value = np.median(block)
        elif method == 'max':
            block_value = np.max(block)
        elif method == 'min':
            block_value = np.min(block)
        else:
            block_value = np.mean(block)

        if block_value > threshold_value:
            output[blocks_y * block_h:, blocks_x * block_w:] = 255
        else:
            output[blocks_y * block_h:, blocks_x * block_w:] = 0

    return output.astype(np.uint8)


def cropping_detection(img, x, y, w, h, threshold_ratio=0.6):
    while True:
        # Left strip (1 x h)
        left_strip = img[y:y+h, x:x+5]
        left_white_ratio = np.sum(left_strip == 255) / left_strip.size

        # Right strip (1 x h)
        right_strip = img[y:y+h, x+w-5:x+w]
        right_white_ratio = np.sum(right_strip == 255) / right_strip.size

        # Decide based on white dominance
        left_white = left_white_ratio > threshold_ratio
        right_white = right_white_ratio > threshold_ratio

        # If both sides are white-dominant, stop
        if left_white and right_white:
            break

        # Crop sides accordingly
        if left_white and not right_white:
            w -= 1  # crop right
        elif right_white and not left_white:
            x += 1
            w -= 1  # crop left
        else:
            x += 1
            w -= 2  # crop both

        # Stop if box becomes too small
        if w <= 1:
            break

    return x, y, w, h


def detect_material_on_bed(empty_bed_path, with_material_path):
    """
    Detect material placed on CNC bed using image comparison

    Args:
        empty_bed_path (str): Path to empty bed image
        with_material_path (str): Path to image with material

    Returns:
        dict: Contains material dimensions, position, and processed images
    """
    if isinstance(empty_bed_path, str):
        empty_bed = cv2.imread(empty_bed_path)
        with_material = cv2.imread(with_material_path)
    else:
        empty_bed = empty_bed_path
        with_material = with_material_path

    if empty_bed is None or with_material is None:
        raise ValueError("Could not load images")

    h1, w1 = empty_bed.shape[:2]
    h2, w2 = with_material.shape[:2]

    if (h1, w1) != (h2, w2):

        min_h, min_w = min(h1, h2), min(w1, w2)
        empty_bed = cv2.resize(empty_bed, (min_w, min_h))
        with_material = cv2.resize(with_material, (min_w, min_h))

    empty_gray = cv2.cvtColor(empty_bed, cv2.COLOR_BGR2GRAY)
    material_gray = cv2.cvtColor(with_material, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(empty_gray, material_gray)
    thresh = block_threshold(diff, (2, 5))
    cv2.imwrite('debug/difference.png', diff)
    cv2.imwrite('debug/difference_thresh.png', thresh)

    contours, _ = cv2.findContours(
                    thresh,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                  )

    min_area = 0
    material_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) >= 4:
                material_contours.append(contour)

    if material_contours:
        largest_contour = max(material_contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)
        x,  y, w, h = cropping_detection(thresh, x, y, w, h)

        result_img = with_material.copy()
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            result_img,
            f'Material: {w}x{h} pixels',
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        return {
            'material_found': True,
            'dimensions': {'width': w, 'height': h},
            'position': {'x': x, 'y': y},
            'area': cv2.contourArea(largest_contour),
            'result_image': result_img,
            'difference_image': diff,
            'threshold_image': thresh,
            'contour': largest_contour
        }

    else:
        return {
            'material_found': False,
            'result_image': with_material,
            'difference_image': diff,
            'threshold_image': thresh
        }


def visualize_detection_process(
        detection_result,
        empty_bed_path,
        with_material_path
):

    """
    Visualize the detection process step by step
    """
    # Load original images for display
    empty_bed = cv2.imread(empty_bed_path)
    with_material = cv2.imread(with_material_path)

    # Convert BGR to RGB for matplotlib
    empty_bed_rgb = cv2.cvtColor(empty_bed, cv2.COLOR_BGR2RGB)
    with_material_rgb = cv2.cvtColor(with_material, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(
                        detection_result['result_image'],
                        cv2.COLOR_BGR2RGB
                    )

    # Create subplot figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original images
    axes[0, 0].imshow(empty_bed_rgb)
    axes[0, 0].set_title('Empty Bed')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(with_material_rgb)
    axes[0, 1].set_title('With Material')
    axes[0, 1].axis('off')

    # Difference image
    axes[0, 2].imshow(detection_result['difference_image'], cmap='gray')
    axes[0, 2].set_title('Difference Image')
    axes[0, 2].axis('off')

    # Threshold image
    axes[1, 0].imshow(detection_result['threshold_image'], cmap='gray')
    axes[1, 0].set_title('Threshold Image')
    axes[1, 0].axis('off')

    # Result with bounding box
    axes[1, 1].imshow(result_rgb)
    axes[1, 1].set_title('Detected Material')
    axes[1, 1].axis('off')

    # Material info
    if detection_result['material_found']:
        pos_x = detection_result['position']['x']
        pos_y = detection_result['position']['y']
        info_text = f"""Material Detected:
    Width: {detection_result['dimensions']['width']} px
    Height: {detection_result['dimensions']['height']} px
    Position: ({pos_x}, {pos_y})
    Area: {detection_result['area']:.0f} pixels²"""
    else:
        info_text = "No material detected"

    axes[1, 2].text(0.1,
                    0.5,
                    info_text,
                    transform=axes[1, 2].transAxes,
                    fontsize=12,
                    verticalalignment='center'
                    )

    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    empty_bed_path = "background_image.png"
    with_material_path = "object_image.png"

    try:
        result = detect_material_on_bed(empty_bed_path, with_material_path)

        if result['material_found']:
            print("Material detected!")
            print(
                f"Dimensions: {result['dimensions']['width']}"
                F"x {result['dimensions']['height']} pixels")
            print(f"Position: ({result['position']['x']}, "
                  F"{result['position']['y']})")
            print(f"Area: {result['area']:.0f} pixels²")
        else:
            print("No material detected")

        visualize_detection_process(result, empty_bed_path, with_material_path)

        cv2.imwrite("detection_result.jpg", result['result_image'])

    except Exception as e:
        print(f"Error: {e}")
