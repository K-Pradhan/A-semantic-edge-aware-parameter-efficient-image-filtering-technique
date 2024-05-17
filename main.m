% clear; clc; close all;
Widx = 3; % Widx = 1/2/3/4

[InputImage] = imread('InputImages/01.jpg');
figure;
imshow(InputImage);
% InputImage = imnoise(InputImage,'gaussian',0,0.05);
InputImage = im2double(InputImage);
[FilteredImage] = AdaptiveWindowApproxFinal(InputImage,Widx); 
figure;
imshow(FilteredImage);