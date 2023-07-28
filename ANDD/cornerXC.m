function [cornerNumber,Cornerimg] = cornerXC(Image)
% ANDD corner detection

if size(Image,3)==3
   Image = rgb2gray(Image);
end

[cornerNumber, Cornerimg,~] = corner_ANDD(Image);
% corners = detectFASTFeatures(Image);
% cornerlocation=corners.Location;
% cornerNumber=corners.Count;
% Cornerimg=zeros(size(Image));
% 
%  for i=1:cornerNumber
%     Cornerimg(cornerlocation(i,2),cornerlocation(i,1))=1;
%  end

end

%detectKAZEFeatures
%mysize=size(rgb);
%if numel(mysize)>2
%  A=rgb2gray(rgb); %将彩色图像转换为灰度图像
%else
%A=rgb;
%end


