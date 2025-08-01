import cv2
import numpy as np


class BackgroundSubtractor:

    def __init__(self, img):
        self.img = img

    def apply_background_subtraction(self):
        # Convert to grayscale and create binary mask for non-black areas
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Close small gaps in the mask
        kernel = np.ones((7, 7), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find the largest contour (main object)
        contours, _ = cv2.findContours(
                        cleaned,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                      )
        largest_contour = max(contours, key=cv2.contourArea)

        # Create exact mask from contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)

        # Apply bounding rectangle crop
        x, y, w, h = cv2.boundingRect(largest_contour)
        roi = self.img[y:y+h, x:x+w]
        mask_roi = mask[y:y+h, x:x+w]

        # Create 4-channel BGRA image with transparent background
        bgra = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = mask_roi  # alpha channel from mask

        # Save result with transparent background
        return bgra
