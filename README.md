# Smart Metal Sheet Detection for Laser Cutting

## Project Overview
This project implements an automated system for measuring 2D objects using computer vision techniques. It's designed specifically for metal sheet detection and measurement in laser cutting applications, featuring camera calibration, image stitching, and precise dimension measurements.

## Features
- Camera calibration using chessboard patterns
- Image undistortion using calibration data
- Multi-image stitching for large objects
- Automated material detection
- Precise measurement conversion from pixels to real-world units
- DXF file generation for detected objects

## Project Structure
```
├── bg_images/              # Background images for different positions
├── calibration_data/       # Camera calibration data files
├── debug/                  # Debug output images and visualizations
├── dimensions/            # Pixel to real-world unit conversion ratios
├── Initialisation/        # Setup and calibration tools
│   ├── Calibration/      # Camera calibration scripts and data
│   └── px_ratio.py       # Pixel to metric conversion utility
├── obj_images/           # Object images for measurement
├── ref_images/           # Reference images for calibration
├── Utils/                # Core utility modules
│   ├── bg_remover.py    # Background subtraction
│   ├── detect_material.py # Material detection algorithms
│   ├── dxf_sketcher.py  # DXF file generation
│   ├── stitcher.py      # Image stitching functionality
│   └── undistorter.py   # Image undistortion using calibration
└── main.py              # Main application script
```

## Setup and Installation

### Prerequisites
- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib

### Initial Setup
1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Camera Calibration
1. Mount cameras in positions with at least 30% overlap between adjacent images
2. Capture chessboard pattern images for each camera position
3. Store calibration images in `Initialisation/Calibration/data/pos_x_y/`
4. Run calibration:
   ```bash
   cd Initialisation/Calibration
   python main.py
   ```
5. Calibration data will be stored in `calibration_data/`

### Pixel to Metric Ratio Setup
1. Place a reference object with known dimensions
2. Update known dimensions in `px_ratio.py`
3. Run the script:
   ```bash
   cd Initialisation
   python px_ratio.py
   ```

## Usage

### Main Application
Run the main application:
```bash
python main.py
```

The application will:
1. Load and undistort background images
2. Stitch multiple images if needed
3. Process object images
4. Detect material and calculate dimensions
5. Generate DXF file of the detected object

### Output Files
- `object_image.png`: Processed object image
- `background_image.png`: Processed background image
- `detected_material.png`: Visualization of detected material
- `rectangle_sketch.dxf`: DXF file of detected object

## Core Components

### Utils Module Details
- `undistorter.py`: Handles camera lens distortion correction using calibration data
  - Loads camera calibration parameters (camera matrix and distortion coefficients)
  - Applies undistortion to remove lens distortions from images
  - Supports batch processing of multiple images in a grid layout
  - Preserves image quality through optimal new camera matrix calculation

- `stitcher.py`: Advanced image stitching implementation
  - Performs custom grid-based image stitching for multiple camera views
  - Handles overlapping regions between adjacent images
  - Supports debug mode for visualization of stitching process
  - Implements image alignment and blending for seamless results

- `detect_material.py`: Sophisticated material detection system
  - Implements background subtraction for material isolation
  - Uses adaptive thresholding for robust material detection
  - Calculates precise dimensions of detected materials
  - Provides visualization tools for detection verification
  - Returns material position, dimensions, and area measurements

- `bg_remover.py`: Background subtraction utility
  - Implements various background removal techniques
  - Handles different lighting conditions
  - Supports dynamic background updates
  - Helps isolate materials from background noise

- `dxf_sketcher.py`: CAD file generation utility
  - Converts detected material measurements to DXF format
  - Creates vector-based outline of detected objects
  - Generates industry-standard CAD files
  - Supports rectangular shape generation with precise dimensions

### Measurement System
- Camera calibration for accurate measurements
- Pixel to metric conversion using reference objects
- Automatic dimension calculation in metric units

### Output Generation
- Visual output for debugging and verification
- DXF file generation for CAD/CAM systems

## Troubleshooting
- Ensure proper camera calibration with good chessboard images
- Verify camera positions have sufficient overlap
- Check lighting conditions for consistent material detection
- Monitor debug output images for processing issues

## Notes
- Camera calibration quality affects measurement accuracy
- Proper lighting is crucial for reliable material detection
- Reference object measurements determine final accuracy
- Debug mode provides detailed visual feedback

## License
[Your License Information Here]

## Contact
[Your Contact Information Here]
