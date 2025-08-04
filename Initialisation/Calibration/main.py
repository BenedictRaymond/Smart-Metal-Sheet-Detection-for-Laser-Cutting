import cv2
from typing import Tuple, List
import numpy as np
import os


class CalibrateCamera:
    def __init__(
        self,
        folder_name: str,
        chessboard_size: Tuple[int, int] = (9, 6)
    ):
        """
        Camera Calibration

        Args:
        folder_name: folder path for storing the calibration results.
        chessboard_size: Size of chessboard pattern.
        """
        self.chessboard_size = chessboard_size
        self.folder_name = folder_name
        self.camera_matrix = None
        self.dist_coeffs = None

    def calibrate_camera(
        self, calibration_images_folder: str = None
    ):

        """
        Calibrate camera intrinsics (focal length, distortion)

        Args:
            calibration_images_folder: Path to calibration images
        """
        print("Camera calibration...")

        if (
            calibration_images_folder
            and
            os.path.exists(calibration_images_folder)
        ):
            for folder in os.listdir(calibration_images_folder):
                folder_path = os.path.join(
                    calibration_images_folder, folder
                )
                if os.path.isdir(folder_path):
                    print(f"Processing folder: {folder}")
                    self._perform_folder_calibration(
                        folder_path
                    )
                    print(f"Camera calibration results of {folder}:")
                    print("Camera matrix:")
                    print(self.camera_matrix)
                    print("Distortion coefficients:")
                    print(self.dist_coeffs.flatten())
                    # Save calibration data
                    self._save_calibration(
                        folder=os.path.basename(folder_path)
                    )
                    # Calculate and display calibration quality metrics
                    self._analyze_calibration_quality(
                            os.path.basename(folder_path)
                    )

                else:
                    print(f"Skipping non-directory: {folder}")
        else:
            # Use default camera parameters or estimate from images
            print(
              "No calibration images provided. Using default camera matrix..."
            )

    def _perform_folder_calibration(
        self,
        folder_path: str
    ):

        """Perform calibration using images from multiple grid positions"""
        print(f"Performing calibration for {os.path.basename(folder_path)}...")

        images = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        print(f"Found {len(images)} calibration images")

        # Perform calibration with all collected images
        self._calibrate_from_image_list(
                image_paths=images,
                folder=os.path.basename(folder_path)
            )

    def _calibrate_from_image_list(
        self,
        image_paths: List[str],
        folder: str
    ):

        """Calibrate camera from a list of image paths"""
        criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001
                    )

        # Prepare object points
        objp = np.zeros(
                    (self.chessboard_size[0] * self.chessboard_size[1], 3),
                    np.float32
                )
        objp[:, :2] = np.mgrid[
                                0:self.chessboard_size[0],
                                0:self.chessboard_size[1]].T.reshape(-1, 2)

        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane
        successful_images = []

        for img_path in image_paths:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(
                                    gray,
                                    self.chessboard_size,
                                    None
                                )

                if ret:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(
                                    gray,
                                    corners,
                                    (11, 11),
                                    (-1, -1),
                                    criteria
                                )
                    imgpoints.append(corners2)
                    successful_images.append(img_path)
                    print(
                     f"✓ Processed: {os.path.basename(img_path)} from {folder}"
                    )
                else:
                    print(
                     f"✗ No chessboard found: {os.path.basename(img_path)}"
                    )
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")

        if len(objpoints) >= 10 or True:  # Minimum images for good calibration
            # Perform calibration
            img_size = gray.shape[::-1]
            (
                ret,
                self.camera_matrix,
                self.dist_coeffs,
                rvecs,
                tvecs
            ) = cv2.calibrateCamera(
                            objpoints,
                            imgpoints,
                            img_size,
                            None,
                            None
                )

            print(f"Camera calibration completed successfully for {folder}!")
            print(f"Used {len(objpoints)} out of {len(image_paths)} images")

            # Calculate reprojection error
            total_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(
                                    objpoints[i],
                                    rvecs[i],
                                    tvecs[i],
                                    self.camera_matrix,
                                    self.dist_coeffs
                                )
                error = cv2.norm(
                            imgpoints[i],
                            imgpoints2,
                            cv2.NORM_L2
                        ) / len(imgpoints2)
                total_error += error

            mean_error = total_error / len(objpoints)

            # Store calibration quality metrics
            self.calibration_metrics = {
                'mean_reprojection_error': mean_error,
                'images_used': len(objpoints),
                'images_total': len(image_paths),
                'success_rate': len(objpoints) / len(image_paths)
            }

    def _analyze_calibration_quality(self, folder: str):
        """Analyze and report calibration quality"""
        if hasattr(self, 'calibration_metrics'):
            metrics = self.calibration_metrics
            mre = metrics['mean_reprojection_error']
            print(f"\nCalibration Quality Assessment of {folder}:")
            print(f"Success rate: {metrics['success_rate']:.1%}")
            print(
             f"Mean reprojection error: {mre:.3f} pixels"
            )

            # Quality assessment
            if metrics['mean_reprojection_error'] < 0.5:
                print("✓ Excellent calibration quality")
            elif metrics['mean_reprojection_error'] < 1.0:
                print("✓ Good calibration quality")
            elif metrics['mean_reprojection_error'] < 2.0:
                print("⚠ Acceptable calibration quality")
            else:
                print("⚠ Poor calibration quality - consider recalibrating")

            print(
             "Focal length:",
             f"\n fx={self.camera_matrix[0, 0]:.1f}",
             f"\n fy={self.camera_matrix[1, 1]:.1f}"
            )
            print(
             "Principal point:",
             f"\n cx={self.camera_matrix[0, 2]:.1f}",
             f"\n cy={self.camera_matrix[1, 2]:.1f}"
            )
            print(
             f"Distortion coefficients: {self.dist_coeffs}",
             f"\nshape: {self.dist_coeffs.shape}"
            )

    def _save_calibration(
        self,
        folder: str
    ):
        """ Save calibration data to a specified folder """

        print(f"Saving calibration data to {self.folder_name}{folder}.npz...")
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
            np.savez(
                os.path.join(self.folder_name, f'{folder}.npz'),
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs
            )
        else:
            np.savez(
                os.path.join(self.folder_name, f'{folder}.npz'),
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs
            )


def main():
    calibrate = CalibrateCamera(r'calibration_data', (9, 6))
    calibrate.calibrate_camera(r'Initialisation\Calibration\data')


if __name__ == '__main__':
    main()
