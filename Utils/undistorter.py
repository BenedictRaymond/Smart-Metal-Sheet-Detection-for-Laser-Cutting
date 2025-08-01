import numpy as np
import cv2
import os


class ImageUndistorter:
    def __init__(self, calibration_data_folder, image_paths, row, col):
        """
        Initialize image undistorter with calibration data from a folder
        """
        self.calibration_data_folder = calibration_data_folder
        self.calibration_data_files = []
        self.row, self.col = (row, col)
        self.image_paths = image_paths
        self.camera_matrix = None
        self.dist_coeffs = None

    def load_calibration_data(self):
        """
        Load camera matrix and distortion coefficients from
        calibration data
        """
        calibration_data_files = [
            os.path.join(self.calibration_data_folder, file)
            for file in os.listdir(self.calibration_data_folder)
        ]

        i = 0
        for _ in range(self.row):
            self.calibration_data_files.append(
                                calibration_data_files[i: i + self.col]
            )
            i += self.row

        return self.undistort_images()

    def undistort_images(self):
        """
        Undistort images using the camera matrix and
        distortion coefficients
        """
        for i in range(len(self.calibration_data_files)):
            for j in range(len(self.calibration_data_files[0])):
                calibration_data = np.load(
                                    self.calibration_data_files[i][j],
                                    'r'
                                    )
                img_p = self.image_paths[i][j]
                cal_p = self.calibration_data_files[i][j]
                print(
                 f"{img_p} is being undistorted by {cal_p}"
                )
                h, w = cv2.imread(self.image_paths[i][j]).shape[:2]
                self.camera_matrix = calibration_data["camera_matrix"]
                self.dist_coeffs = calibration_data["dist_coeffs"]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                                        self.camera_matrix,
                                        self.dist_coeffs,
                                        (w, h),
                                        1,
                                        (w, h)
                                    )
                self.image_paths[i][j] = cv2.undistort(
                                            cv2.imread(self.image_paths[i][j]),
                                            self.camera_matrix,
                                            self.dist_coeffs,
                                            None,
                                            newcameramtx
                                    )
                x, y, w, h = roi
                self.image_paths[i][j] = self.image_paths[i][j][y:y+h, x:x+w]
                cv2.imwrite(
                    f"debug\\undistorted_images_img_{i}_{j}.jpg",
                    self.image_paths[i][j]
                )

        return self.image_paths


if __name__ == "__main__":
    # Example usage
    image_paths = [
        [
            r"grid_images\img_0_0.jpg",
            r"grid_images\img_0_1.jpg"
        ],
        [
            r"grid_images\img_1_0.jpg",
            r"grid_images\img_1_1.jpg"
        ]
    ]
    undistorter = ImageUndistorter(r"calibration_data", image_paths)
    undistorter.load_calibration_data()
    for i in range(len(image_paths)):
        for j in range(len(image_paths)):
            cv2.imwrite(
                f"debug\\undistorted_images_img_{i}_{j}.jpg",
                undistorter.image_paths[i][j]
            )
