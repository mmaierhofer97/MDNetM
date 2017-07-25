function sim = ch_sim(Im1,Im2)
%Im1 = imread('000001.jpg');
%Im2 = imread('000002.jpg');

%  convert images to type double (range from from 0 to 1 instead of from 0 to 255)
Im1 = im2double(Im1);
Im2 = im2double(Im2);

%Im1=zeros(3,3,3)
%Im2=ones(3,3,3)

Red1 = Im1(:, :, 1);
Green1 = Im1(:, :, 2);
Blue1 = Im1(:, :, 3);
HnBlue1 = imhist(Blue1)./numel(Blue1);
HnGreen1 = imhist(Green1)./numel(Green1);
HnRed1 = imhist(Red1)./numel(Red1);

Red2 = Im2(:, :, 1);
Green2 = Im2(:, :, 2);
Blue2 = Im2(:, :, 3);
HnBlue2 = imhist(Blue2)./numel(Blue2);
HnGreen2 = imhist(Green2)./numel(Green2);
HnRed2 = imhist(Red2)./numel(Red2);

%hn1 = imhist(Im1(:))./numel(Im1);
%hn2 = imhist(Im2(:))./numel(Im2);
% Calculate the histogram error
sim = 1 - mean([sum(imabsdiff(HnBlue1,HnBlue2)),sum(imabsdiff(HnRed1,HnRed2)),sum(imabsdiff(HnGreen1,HnGreen2))])/2;

