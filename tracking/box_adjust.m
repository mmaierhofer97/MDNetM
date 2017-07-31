function [ resultLoc ] = box_adjust(image, targetLoc)
imi = max(targetLoc(1), 1);
jmi = max(targetLoc(2), 1);
ima = min(targetLoc(1) + targetLoc(3), size(image,2) - 1);
jma = min(targetLoc(2) + targetLoc(4), size(image,1) - 1);
resultLoc(1) = imi;
resultLoc(2) = jmi;
resultLoc(3) = ima - imi;
resultLoc(4) = jma - jmi;
end