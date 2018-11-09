clear
clc

img  = imread('peppers.png');
img_orig = double(img(:,:,2));
sig = 1.5;
t = fspecial('gaussian',[7,7],sig);
img_blurred = imfilter(img_orig,t);

[k] = cal_kernel(img_orig,img_blurred,[7,7],1);

figure; imshow([t,k],[])
title ( 'estimated kernel,truth kernel');
set(gcf,'unit','centimeters','position',[10 5 20 10]);