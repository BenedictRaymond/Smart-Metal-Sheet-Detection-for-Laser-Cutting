import cv2
import numpy as np
import os
from typing import List, Tuple, Optional


class CustomPanoramaStitcher:
    def __init__(
            self,
            template_width_ratio=0.3,
            search_width_ratio=0.5,
            overlap_threshold=0.8
    ):
        """
        Initialize custom panorama stitcher using template matching

        Args:
            template_width_ratio: Ratio of image width to use as template
            search_width_ratio: Ratio of right side of first image to search in
            overlap_threshold: Threshold for considering a match good
        """
        self.template_width_ratio = template_width_ratio
        self.template_width = 0
        self.search_width_ratio = search_width_ratio
        self.search_width = 0
        self.overlap_threshold = overlap_threshold
        self.count = 0
        self.current_row = 0
        self.current_col = 0
        self.initial = True

    def extract_template(self, img: np.ndarray) -> np.ndarray:
        """Extract left portion of image as template"""
        return img[:, :self.template_width]

    def calculate_matching_score(
            self,
            template: np.ndarray,
            region: np.ndarray
    ) -> Tuple[float, dict]:
        """
        Calculate matching score between template and region
        using multiple metrics

        Returns a score between 0 and 1 (1 = perfect match) and debug info
        """
        if template.shape != region.shape:
            return 0.0, {"error": "Shape mismatch"}

        # Convert to grayscale for consistent processing
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template
            region_gray = region

        # Method 2: Normalized Cross Correlation
        ncc_result = cv2.matchTemplate(
                        region_gray,
                        template_gray,
                        cv2.TM_CCOEFF_NORMED
                    )
        ncc_score = ncc_result[0, 0]
        ncc_score = max(0, ncc_score)  # Ensure non-negative

        debug_info = {
            "ncc_score": ncc_score,
        }

        return ncc_score, debug_info

    def find_best_match_position(
            self,
            img1: np.ndarray,
            img2: np.ndarray
    ) -> Tuple[int, float]:
        """
        Find the best horizontal position to place img2 relative to img1
        Template (left 30% of img2) slides over right 50% of img1

        Returns:
            Tuple of (best_x_position, best_score)
        """
        # Extract template from left side of img2
        template = self.extract_template(img2)
        template_height, template_width = template.shape[:2]

        img1_height, img1_width = img1.shape[:2]

        # Define search region (right 50% of img1)
        search_start_x = img1_width - self.search_width
        search_region_width = self.search_width

        # Ensure template fits in img1 height
        if template_height > img1_height:
            # Resize template to fit img1 height
            scale_factor = img1_height / template_height
            new_width = int(template_width * scale_factor)
            template = cv2.resize(template, (new_width, img1_height))
            template_height, template_width = template.shape[:2]

        # Ensure template fits in search region width
        if template_width > search_region_width:
            print(
             (f"Template width {template_width} exceeds"),
             (f"search region width {search_region_width}"))
            return 0, 0.0

        best_score = -1.0  # Start with impossible score
        best_x = search_start_x
        all_scores = []  # Store all scores for analysis

        # Slide template horizontally across right 50% of img1
        for x in range(search_start_x, img1_width - template_width + 1):
            # Extract region from img1 at current position
            region = img1[:template_height, x:x + template_width]

            # Calculate matching score with debug info
            score, debug_info = self.calculate_matching_score(template, region)
            all_scores.append((x, score, debug_info))

            if score > best_score:
                best_score = score
                best_x = x

        return best_x, best_score

    def stitch_images_at_position(
            self,
            img1: np.ndarray,
            img2: np.ndarray,
            overlap_x: int
    ) -> np.ndarray:
        """
        Stitch img2 to img1 at the specified overlap position
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        temp_width_ratio = int(w2 * self.template_width_ratio)
        overlap_width = temp_width_ratio

        img2_start_x = overlap_x + overlap_width - temp_width_ratio

        final_width = max(w1, img2_start_x + w2)
        final_height = max(h1, h2)

        result = np.zeros((final_height, final_width, 3), dtype=np.uint8)

        result[:h1, :w1] = img1

        blend_start = max(0, img2_start_x)
        blend_end = min(w1, img2_start_x + w2)

        if blend_start < blend_end:
            overlap_width_actual = blend_end - blend_start

            alpha = np.linspace(1, 0, overlap_width_actual)

            for i in range(overlap_width_actual):
                x_pos = blend_start + i
                img2_x = x_pos - img2_start_x

                if 0 <= img2_x < w2:
                    result[:min(h1, h2), x_pos] = (
                        alpha[i] * result[:min(h1, h2), x_pos] +
                        (1 - alpha[i]) * img2[:min(h1, h2), img2_x]
                    )

        non_overlap_start = max(img2_start_x, blend_end)
        non_overlap_end = img2_start_x + w2

        if non_overlap_start < non_overlap_end:
            img2_start_offset = non_overlap_start - img2_start_x
            result[
                :h2,
                non_overlap_start:non_overlap_end
            ] = img2[:, img2_start_offset:]

        return result

    def stitch_pair(
            self,
            img1: np.ndarray,
            img2: np.ndarray
    ) -> Optional[np.ndarray]:

        """Stitch two images together using custom template matching"""
        self.count += 1

        if img1 is None or img2 is None:
            print("One or both images are None")
            return None

        print(f"\n=== Stitching pair {self.count} ===")

        if self.initial:
            best_x, best_score = self.find_best_match_position(img1, img2)
            filename = f"{self.current_row}.txt"

            # Create directory if it doesn't exist
            if not os.path.exists('overlap_x_values'):
                os.makedirs('overlap_x_values')

            filepath = os.path.join('overlap_x_values', filename)

            with open(filepath, 'w') as f:
                f.write(f"{best_x}\n")

        else:
            with open(fr'overlap_x_values\{self.current_row}.txt', 'r') as f:
                best_x = int(f.read().strip())

        result = self.stitch_images_at_position(img1, img2, best_x)

        cv2.imwrite(f"debug/result_{self.count}.jpg", result)

        self.current_row += 1
        # Create visualization of the match
        template = self.extract_template(img2)
        visualization = self.visualize_match(img1, img2, best_x, template)
        cv2.imwrite(
                f"debug/match_visualization_{self.count}.jpg",
                visualization
        )

        return result

    def visualize_match(
            self,
            img1: np.ndarray,
            img2: np.ndarray,
            match_x: int,
            template: np.ndarray
    ) -> np.ndarray:
        """Create visualization showing the matching region and search area"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        th, tw = template.shape[:2]

        # Create side-by-side visualization
        vis_height = max(h1, h2)
        vis_width = w1 + w2 + 20  # 20 pixel gap
        visualization = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)

        # Place images
        visualization[:h1, :w1] = img1
        visualization[:h2, w1+20:w1+20+w2] = img2

        # Draw search region (right 50% of img1) in blue
        search_start_x = w1 - self.search_width
        cv2.rectangle(
            visualization,
            (search_start_x, 0),
            (w1, h1),
            (255, 0, 0),
            2
        )

        # Draw rectangle around matching region in img1 (green)
        cv2.rectangle(
            visualization,
            (match_x, 0),
            (match_x + tw, th),
            (0, 255, 0),
            3
        )

        # Draw rectangle around template in img2 (green)
        cv2.rectangle(
            visualization,
            (w1+20, 0),
            (w1+20+tw, th),
            (0, 255, 0),
            3
        )

        # Add text labels
        cv2.putText(
            visualization,
            f"Match at x={match_x}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        cv2.putText(
            visualization,
            f"Search region (right {int(self.search_width_ratio*100)}%)",
            (search_start_x, h1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )
        cv2.putText(
            visualization,
            f"Template (left {int(self.template_width_ratio*100)}%)",
            (w1+20, h2-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        return visualization

    def stitch_multiple(
            self,
            images: List[np.ndarray]
    ) -> Optional[np.ndarray]:

        """Stitch multiple images together"""
        if len(images) < 2:
            return images[0] if images else None

        h, w = images[0].shape[:2]
        self.template_width = int(w * self.template_width_ratio)
        self.search_width = int(w * self.search_width_ratio)

        result = self.stitch_pair(images[0], images[1])

        if result is None:
            return None

        # Stitch remaining images
        for i in range(2, len(images)):
            result = self.stitch_pair(result, images[i])
            if result is None:
                print(f"Failed to stitch image {i}")
                return None

        return result


class CustomGridImageStitching:
    def __init__(
            self,
            images: List,
            grid_size: Tuple[int, int] = (3, 3),
            debug: bool = False,
            initial_process: bool = True
    ):

        """
        Custom grid image stitching using template matching

        Args:
            images: List of images in grid format
            grid_size: Size of the grid (rows, cols)
            debug: Whether to save debug images
        """
        self.grid_rows, self.grid_cols = grid_size
        self.images = images
        # Initialize with custom parameters: 30% template, 50% search region
        self.stitcher = CustomPanoramaStitcher(template_width_ratio=0.18,
                                               search_width_ratio=0.36,
                                               overlap_threshold=0.8)
        self.debug = debug
        self.initial_process = initial_process
        self.stitcher.initial = self.initial_process

    def process_stitching(self):
        """Stitch images in grid method using custom template matching"""

        total_images = len(self.images) * len(self.images[0])
        if self.grid_rows * self.grid_cols != total_images:
            print("Not sufficient Images.")
            return None

        row_stitched = []

        for i, row_images in enumerate(self.images):
            print(f"\nProcessing row {i}")

            if isinstance(row_images[0], str):
                row_images = [cv2.imread(path) for path in row_images]

            row_images = [img for img in row_images if img is not None]

            if not row_images:
                print(f"No valid images in row {i}")
                continue

            stitched_row = self.stitcher.stitch_multiple(row_images)

            if stitched_row is None:
                print(f"Failed to stitch row {i}")
                continue

            row_stitched.append(stitched_row)

            if self.debug:
                cv2.imwrite(f"debug/row_{i}_stitched.jpg", stitched_row)

        if not row_stitched:
            print("All row stitching failed")
            return None

        print("Stitching rows together...")
        result = self.stitcher.stitch_multiple(row_stitched)

        if result is None:
            print("Failed to stitch columns")
            return None
        self.stitcher.current_col += 1

        return result
