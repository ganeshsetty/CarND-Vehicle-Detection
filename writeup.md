### Histogram of Oriented Gradients (HOG)


#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
* Read training images from vehicles and non-vehicles folders.

![](./output_images/car_imgs.png)
![](./output_images/notcar_imgs.png)

* Extract HOG features and visualized the HOG images
![](./output_images/hog_orient_car.png)



#### 2. Explain how you settled on your final choice of HOG parameters.

* Color Histograms for RGB, YCrCb, HSV

Different color spaces(YCrCb,HSV,RGB) configuration

![](./output_images/hog_colorspace_car.png)

Different orientations,pix per cell, cells per block parameters configuration for YCrCb color space

* Orientations = 9, Pixel per cell = (8,8), cells per block = 2  
* Orientations = 8, Pixel per cell = (8,8), cells per block = 2
* Orientations = 9, Pixel per cell = (16,16), cells per block = 2
* Orientations = 8, Pixel per cell = (16,16), cells per block = 2
* Orientations = 9, Pixel per cell = (8,8), cells per block = 1
* Orientations = 8, Pixel per cell = (8,8), cells per block = 1

![](./output_images/hog_YCrCb_orient_cells_per_block.png)


