from Utils.stitcher import CustomGridImageStitching
from Utils.undistorter import ImageUndistorter
# from Utils.bg_remover import BackgroundSubtractor
import Utils.detect_material as detect_material
from Utils.dxf_sketcher import create_dxf
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    obj_images = [
        [
            r"ref_images\dim_y_large_01.png",
            r"ref_images\dim_y_large_02.png",
            r"ref_images\dim_y_large_03.png"
        ]
    ]

    bg_images = [
        [
            r"bg_images\pos_1.png",
            r"bg_images\pos_2.png",
            r"bg_images\pos_3.png"
        ]
    ]

    path = 'calibration_data'
    undistorter = ImageUndistorter(path, bg_images, row=1, col=3)
    bg_images = undistorter.load_calibration_data()

    grid_stitcher = CustomGridImageStitching(bg_images, (1, 3), debug=True)
    bg_image = grid_stitcher.process_stitching()

    undistorter = ImageUndistorter(path, obj_images, row=1, col=3)
    obj_images = undistorter.load_calibration_data()

    grid_stitcher = CustomGridImageStitching(
                        obj_images,
                        (1, 3),
                        debug=True,
                        initial_process=False)
    obj_image = grid_stitcher.process_stitching()

    cv2.imwrite("object_image.png", obj_image)
    cv2.imwrite("background_image.png", bg_image)

    try:
        is_found = detect_material.detect_material_on_bed(bg_image, obj_image)
        if is_found['material_found']:
            print("Material Found")
            with open(r"dimensions\height_ratio.txt", "r") as f:
                height_ratio = f.read().strip()
            with open(r"dimensions\width_ratio.txt", "r") as f:
                width_ratio = f.read().strip()
            h_mm = is_found['dimensions']['height'] / float(height_ratio)
            w_mm = is_found['dimensions']['width'] / float(width_ratio)

            # Round the dimensions
            h_mm = round(h_mm, 1)
            w_mm = round(w_mm, 1)

            # Create figure with 1x3 subplot grid
            plt.figure(figsize=(15, 5))

            # Plot background image
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB))
            plt.title('Background Image')
            plt.axis('off')

            # Plot object image
            plt.subplot(132)
            plt.imshow(cv2.cvtColor(obj_image, cv2.COLOR_BGR2RGB))
            plt.title('Object Image')
            plt.axis('off')

            # Plot detected material
            plt.subplot(133)
            plt.imshow(
                    cv2.cvtColor(is_found['result_image'], cv2.COLOR_BGR2RGB)
            )
            plt.title('Detected Material')
            plt.axis('off')

            # Add dimensions as text below the plots
            plt.figtext(
                0.5,
                0.02,
                f'Dimensions: {h_mm}m x {w_mm}m',
                ha='center',
                fontsize=12,
                bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5}
            )

            # Save the plot
            plt.savefig(
                'measurement_results.png', bbox_inches='tight', dpi=300
            )
            plt.show()

            if create_dxf(h_mm, w_mm, filename="rectangle_sketch.dxf"):
                print("DXF file created successfully.")
            else:
                print("Failed to create DXF file.")
            cv2.imwrite("detected_material.png", is_found['result_image'])
        else:
            print('Material Not Found')

    except Exception as e:
        print(f"Error: {e}")
