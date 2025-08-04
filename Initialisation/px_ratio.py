import os
import sys
import cv2

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Utils.stitcher import CustomGridImageStitching  # noqa: E402
from Utils.undistorter import ImageUndistorter  # noqa: E402
import Utils.detect_material as detect_material  # noqa: E402

ref_images = [
    [
        r"ref_images\dim_large_01.png",
        r"ref_images\dim_large_02.png",
        r"ref_images\dim_large_03.png"
    ]
]

path = 'calibration_data'
undistorter = ImageUndistorter(path, ref_images, row=1, col=3)
ref_images = undistorter.load_calibration_data()

grid_stitcher = CustomGridImageStitching(
                        ref_images,
                        (1, 3),
                        debug=True,
                        initial_process=False
                )

ref_image = grid_stitcher.process_stitching()
cv2.imwrite(
    r"E:\TeckInsoft\Automated_Measurements_2D\debug\reference_image.png",
    ref_image
)

known_height = 1
known_width = 2.5

bg_image = cv2.imread(
                r'E:\TeckInsoft\Automated_Measurements_2D\background_image.png'
            )

try:
    is_found = detect_material.detect_material_on_bed(bg_image, ref_image)
    if is_found['material_found']:
        px_height = is_found['dimensions']['height']
        px_width = is_found['dimensions']['width']

        height_ratio = px_height / known_height
        width_ratio = px_width / known_width
        print("Material Found")
        print(f"Height Ratio: {height_ratio}px")
        print(f"Width Ratio: {width_ratio}px")
        cv2.imwrite("detected_material.png", is_found['result_image'])
        h_filename = "height_ratio.txt"
        w_filename = "width_ratio.txt"

        # Create directory if it doesn't exist
        if not os.path.exists(
            r'E:\TeckInsoft\Automated_Measurements_2D\dimensions'
        ):
            os.makedirs(r'E:\TeckInsoft\Automated_Measurements_2D\dimensions')

        h_filepath = os.path.join(
            r'E:\TeckInsoft\Automated_Measurements_2D\dimensions', h_filename)
        w_filepath = os.path.join(
            r'E:\TeckInsoft\Automated_Measurements_2D\dimensions', w_filename)

        with open(h_filepath, 'w') as f:
            f.write(f"{height_ratio}\n")
        with open(w_filepath, 'w') as f:
            f.write(f"{width_ratio}\n")
    else:
        print('Not Found')

except Exception as e:
    print(f"Error: {e}")
