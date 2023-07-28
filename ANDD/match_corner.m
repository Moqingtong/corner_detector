function [missed_corner, false_corner, local,detected_num,standard_num] = match_corner(detected_corner_image, standard_corner_image)
%验证角点检测器检测角点与标准角点进行对比。
%
%-----------------------
detected_num = sum(sum(detected_corner_image));
standard_num = sum(sum(standard_corner_image));
%-----------------------
[rows,cols] = size(detected_corner_image);
[rows_s,cols_s] = size(standard_corner_image);
if rows_s ~= rows || cols_s ~= cols
    error('images size do not consist');
end
%-----------------------
width = 4;
tmp_detected_corner_image = zeros(rows+2*width,cols+2*width);
tmp_detected_corner_image(width+1:end-width,width+1:end-width) = detected_corner_image;

match_num = 0;
d2 = [];

for i = 1:rows
    for j = 1:cols
        if 1 == standard_corner_image(i,j)
            if sum(sum(tmp_detected_corner_image(i:i+2*width,j:j+2*width))) > 0
                [r,c] = find(tmp_detected_corner_image(i:i+2*width,j:j+2*width) == 1);
                dist = (r-width-1).^2 + (c-width-1).^2;
                [min_dist, index] = min(dist);
                d2 = [d2; min_dist];
                point = [i, j] + [r(index),c(index)] - 1;
                tmp_detected_corner_image(point(1), point(2)) = 0;
                standard_corner_image(max(1,i-1):min(i+1,rows),max(1,j-1):min(j+1,cols)) = 0;                
                match_num = match_num+1;
            end
        end
    end
end
a = match_num;
local = sqrt(sum(d2)/a);

missed_corner = standard_num - match_num;
false_corner = detected_num - match_num;