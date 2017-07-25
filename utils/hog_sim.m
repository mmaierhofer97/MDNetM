function sim = hog_sim(Im1,Im2)
%Im1 = imread('000001.jpg');
%Im2 = imread('000002.jpg');

Im2 = imresize(Im2,[size(Im1,1),size(Im1,2)]);
features1 = extractHOGFeatures(Im1,'CellSize',[16 16]);
features2 = extractHOGFeatures(Im2,'CellSize',[16 16]);

%sim = sum((features1-features2).^2)
sim = cosine_sim(features1,features2);