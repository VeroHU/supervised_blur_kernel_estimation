# supervised_blur_kernel_estimation
If you have the original image and the blurred image, you can use this code to estimate the blur kernel.

This code is modified from:
https://www.mathworks.com/matlabcentral/fileexchange/54944-calculate-blur-kernel-from-original-and-blurry-images
However, our method is based on LEAST SQUARE FIT, which is more accurate than the original code.

please run "demo_main.m" to see the results.


results:
![image](https://github.com/VeroHU/supervised_blur_kernel_estimation/blob/master/peppers.jpg)
![image](https://github.com/VeroHU/supervised_blur_kernel_estimation/blob/master/kernel.jpg)
