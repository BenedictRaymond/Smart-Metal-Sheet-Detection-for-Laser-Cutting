# Smart Metal Sheet Detection for Laser Cutting

## Initialisation Steps

### Calibration
1. Define the positions of the camera to be mounted (Need to be atleast 30% of each image to be overlapped in neighbor images)
2. Then Calibrate each camera with chechkerboard images as data stored respective to their positions in '''Initialisation\Calibration\data''' Folder
3. Calibrated Camera's data should be stored in '''calibration_data''' folder.

### Determining pixel to metre ratio
Note: Actually we can achieve thhis by using a calibration checkerboard but the thing is we need a bigger size and shile stitching the image some anomalies will happen this would create unnecessary problems. So, I decided to use a reference object which will be better for our usecase.

1. Take an object with known dimmensions.
2. Change the variables in '''px_ratio.py''' with known dimensions. Then run it.

