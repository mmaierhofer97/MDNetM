function b = full_over(rect1, rect2)
inter_area = rectint(rect1,rect2);
b=inter_area/min([rect1(3)*rect1(4),rect2(3)*rect2(4)]);
end