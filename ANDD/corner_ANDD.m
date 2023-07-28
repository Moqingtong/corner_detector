function [corner_num, marked_img,classification] = corner_ANDD(varargin)
% =========================================================================
% This is an implementation of the algorithm in the paper "Corner Detection
% and Classification Using Anisotropic Directional Derivative Representations"
% on IEEE Transaction Image Processing.
%
% Kindly report any suggestions or corrections to plshui@xidian.edu.cn
%
%-------------------------------------------------------------------------
% The outline of the proposed corner detection and classification algorithm is:
% 1.	 Extract a binary edge map from the input image by Canny edge detector.
% 2.	 Extract edge contours from the edge map and select the long edges,
%        fill the small gap between two end point and stretch the end point 
%        to the nearest edge pixel with their distance smaller than a fixed size.
% 3.	 Smooth the pixels on the edge contours using ANDD filters, compute 
%        the corner measure for each pixel on the contour.
% 4.	 Apply the non-maximum suppression to the the candidate corners
%        obtained in step 3 within a window. 
%        On the contour, if the corner of one pixel is not only a local maxima
%        in its neighborhood with a given size,the pixel but also above the 
%        previously specified threshold,it is marked as the candidate corner 
%        and the NESMED of the pixel is kept,otherwise, it would be set to zero.
% 5.     Classify the detected corners, For each detected corner,get its
%        ANDD representation by more orientation, and classify the corner by 
%        finding the peak number of the ANDD representation
% -------------------------------------------------------------------------
% Syntax :
% Input :
%  (1)im           - the input image, it could be gray, color or binary image.
%                    If im is empty([]), input image can be get from a open 
%                    file dialog box.
%  (2)eta          - denote the threshhold of the corner measure.The default
%                    value is 0.12.
%  (3)rho2         - denote the square of anisotropic factor for
%                    convenience.The default value is 6.0.
%  (4)canny_thresh - denote the threshold of Canny edge detector respectively.
%                    The default value is [0.10,0.35].
%  (5)sigma        - denate the gaussian scale be used to the Canny edge
%                    detector.The default value is 1.0.
%  (6)isClassifying -denote the flag which determines whether to classify the
%                    detected corners or not.Values 1 means classifying,value
%                    0 means not.The default value is 1.
% 
% Output :
%   (1)corner_num   -  the detected corner number
%   (2)marked_img   - image with detected corner marked.
%   (3)classification- The corner classification result.In each row, the first 
%                     two values indicate a detected corner location and the last
%                     value is the corner type flag corresponding the corner.
%                     Value 2 means a simple corner,value 3 means a Y-type corner
%                     and it means a an X-type or star-like corner when the
%                     value is larger than 3.
% -------------------------------------------------------------------------
% Basic Usage: 
%   Given a test image whose dynamic range is 0-255.
%
%   e.g. 
%   im = imread('block.gif');
%   [corner_num, marked_img] = corner_ANDD(im);
%   
% Advanced Usage:
%   User defined parameters. For example:
%   eta = 0.12; rho2 = 6.0; canny_thresh =[0.10 0.35]; 
%   sigma = 1; isClassifying = 0;
%   im = imread('block.gif');
%   [corner_num, marked_img] = corner_ANDD(im,eta,rho2,canny_thresh,sigma,isClassifying);
%
% Visualize the results:
%   corner_num             % Gives the number of the detected corners
%   [r1,c1] = find(marked_img>0);
%   figure;imshow(im,[]); hold on;
%   plot(c1(:),r1(:),'rs','LineWidth',1,'MarkerEdgeColor','k', 'MarkerSize',4)
%                          % Mark the points on the input image
% -------------------------------------------------------------------------
%   Composed by Peng-Lang Shui
%       Xian,China,Jan.2012
%   Algorithm is derived from:
%       Peng-Lang Shui,Wei-Chuan Zhang,"Corner Detection and Classification 
%   Using Anisotropic Directional Derivative Representations" on IEEE 
%   Transaction Image Processing. 
% =========================================================================

% ------------------ initialize the parameter -----------------------------
[im,eta,rho2,canny_thresh,sigma,isClassifing]=parse_inputs(varargin{:});
if 3 == size(im,3) 
    im = rgb2gray(im);  % Transform RGB image to a Gray one.
end
im = double(im);
[rows cols] = size(im); % Get the size of the extended image
P           = 84;       % The number of the orientations used to detect the corner,
%                         The default value is 84
Gap_size    = 1;        % a paremeter to be used to fill the gaps in the contours, the gap
%                         not more than gap_size were filled in this stage. The default 
%                         Gap_size is 1 pixels
radius_nonmaxsuppt = 5; % the width of the window for the non-maximum suppressionIt ,ranges
%                         from 3 to 6 and the default value is 5
% ---------------------- Image Extension ----------------------------------
width = 10;             % The extending area width of image,and the width of the anisotropic
%                         Directional derivative filter window will be 2*width+1
extend_image = padarray(im,[width,width],'symmetric');
ANDD_FilterBank=anisotropic_Directional_derivative_filter(rho2,rho2,P,width);
%                       % Get the ANDD filters
% ------------------ Contour Extraction and Patching-----------------------
% Apply Canny edge detection to the gray-level image
Extracted_extend_edge = double(edge(extend_image,'canny',canny_thresh,sigma));
% Extract the curves from the edge map and fill the small gap and Stretch the open contours
Extracted_edge=zeros(size(extend_image));
Extracted_edge(width+1:rows+width,width+1:cols+width)=Extracted_extend_edge(width+1:rows+width,width+1:cols+width);
[curve, curve_start, curve_end, curve_mode, curve_num,BW_edge] = extract_curve(Extracted_edge,Gap_size);
% --------------------Corner Decision Via Residual Area--------------------
% Computation of corner measure
max_id=zeros(size(extend_image));   % The orientation index of the maximum of filtering results
R_Area = zeros(size(extend_image)); % The residual area of all edge pixels
for curve_id = 1:curve_num
    edgept_num=length(curve{curve_id});
    tem_curve=curve{curve_id};
    template=[];
    edge_dir=zeros(edgept_num,P);
    Area_edge=zeros(1,edgept_num);

    stren_ve=[1 0];                 % The strength of a step edge
    for i=1:edgept_num
        for direction=1:P
           template(i, direction) =sum(sum(extend_image(tem_curve(i,1)-width:tem_curve(i,1)+width,...;
               tem_curve(i,2)-width:tem_curve(i,2)+width).*ANDD_FilterBank(:,:,direction))); 
        end
        % Normalising the magnitudes
        template(i, :)=template(i, :)/max(abs(template(i, :)));
       
        max_id(tem_curve(i,1),tem_curve(i,2))=find(template(i,:)==max(template(i,:)));
        max_i=(max_id(tem_curve(i,1),tem_curve(i,2))-1)/P*2*pi;
        % Calculate ANDD representation of the edge corresponding the direction of the
        % maximal magnitude
        angle_ve=[max_i,max_i+pi];
        edge_dir(i,:)=ANDDofUCM(rho2,rho2,stren_ve,angle_ve,P);
        edge_dir(i,:)=edge_dir(i,:)/max(abs(edge_dir(i,:)));
        % Calculate the Residual Area
        Area_edge(i)=pi/P*sum((template(i, :)-edge_dir(i,:)).^2);
        R_Area(tem_curve(i,1),tem_curve(i,2)) =Area_edge(i);
    end
    Area{curve_id} = Area_edge;
end

% Non-maximum suppression And Thresholding
[r,c] = nonmaxsuppts(R_Area,radius_nonmaxsuppt,eta);
% -------------------- Corner Detection Result ----------------------------
% Output the number of the detected corners and a image mask with the
% value 1 to the location of corner and value 0 to other location
marked_img = zeros(size(im));
for i=1:size(r,1)
    marked_img(r(i)-width,c(i)-width)=1;
end
corner_num=sum(marked_img(:));
%---------------------Show The Marked Corner------------------------------
% [r1,c1] = find(marked_img>0);
% figure;imshow(im,[]);
% hold on;plot(c1(:),r1(:),'rs','LineWidth',1,'MarkerEdgeColor','k', 'MarkerSize',4)
% --------------------  Corner Classification -----------------------------
if isClassifing
    P_classify  = 384;  % The number of the orientations used to classify corner 
%                         type The default value is 384
    sigma = 4*4;        % The scale factor used to classification
    rho   = 10;         % The anisotropic factor used to classification
    corner_index = [r,c]; 
    ANDD_FilterBank=anisotropic_Directional_derivative_filter(sigma,rho,P_classify,width);
    classification = corner_classify(corner_index,extend_image,ANDD_FilterBank,eta);
end

%%%%%%%%%%%%%% The sub-functions used in the main function  %%%%%%%%%%%%%%%
% =========================================================================
function anigs_direction=anisotropic_Directional_derivative_filter(sigma1,rho1,p,width)
% This fnction is used to compute the anisotropic Directional derivative filters
% -------------------------------------------------------------------------
% Input :
%   (1)sigma1    -- The square of the scale factor.
%   (2)rho1      -- The square of the anisotropic factor.
%   (3)p         -- The orientation number of the anisotropic.
%                   Directional derivative filters.
%   (4)width     -- the width of the anisotropic Directional derivative filter
%                   window will be 2*width+1.
% Output:
%   anigs_direction  -- the generated anisotropic Directional derivative
%                       filters with the size [2*width+1,2*width+1,p],and
%                       each plane indicates a filter with a fixed orientation
% -------------------------------------------------------------------------
anigs_direction(1:(2*width+1),1:(2*width+1),p) = 0;

for direction = 1:p
    theta = (direction-1)*2*pi/p;
    for x = -width:1:width
        for y = -width:1:width
            xr = x*cos(theta)+y*sin(theta);
            yr = -x*sin(theta)+y*cos(theta);
         
            anigs_direction(x+width+1,y+width+1,direction) = ...
            -rho1*xr*1/(2*pi*sigma1^2)*exp(-1/(2*sigma1)*(xr^2*rho1 + yr^2/rho1));
    
        end
    end
end

% =========================================================================   
function [curve,curve_start,curve_end,curve_mode,cur_num,BW_edge]=extract_curve(BW,Gap_size)
%   This function is used to extract curves from binary edge map, if the endpoint of a
%   contour is nearly connected to another endpoint, fill the gap and continue
%   the extraction. 
% -------------------------------------------------------------------------
% Input:
%   (1)BW          -- The obtained edge map with value 1 corresponding to edge pixel and
%                     value 0 corresponding to non-edge pixel.
%   (2)Gap_size    -- The paremeter to be used to fill the gaps in the contours, the gap
%                     not more than gap_size were filled in this stage. The default 
%                     Gap_size is 1 pixels.
% Output:
%   (1)curve       -- The extracted curves whose each element is an individual chain 
%                     code corresponding a contour.
%   (5)cur_num     -- The number of the contours.
%   (6)BW_edge     -- The edge map after filling the small gap and Stretch the open contours
% -------------------------------------------------------------------------
[L,W]=size(BW);
BW1=zeros(L+2*Gap_size,W+2*Gap_size);
BW_edge=zeros(L,W);
% The centre part of the zero matrix BW1 is replaced by the binary edge image BW. 
% The extended matrix makes it convenient to fill the gaps;
BW1(Gap_size+1:Gap_size+L,Gap_size+1:Gap_size+W)=BW;
[r,c]=find(BW1==1);
cur_num=0;
curve = {};

while size(r,1)>0                 % whether the whole image is tranversed
    point=[r(1),c(1)];            % the first edge point coordinate
    cur=point;
    BW1(point(1),point(2))=0;
    [I,J]=find(BW1(point(1)-Gap_size:point(1)+Gap_size,point(2)-Gap_size:point(2)+Gap_size)==1);
    while size(I,1)>0             % if there is at least one edge point around the original point, go on;
        dist=(I-Gap_size-1).^2+(J-Gap_size-1).^2;
        [min_dist,index]=min(dist);
        point=point+[I(index),J(index)]-Gap_size-1;
        cur=[cur;point];
        BW1(point(1),point(2))=0;
        [I,J]=find(BW1(point(1)-Gap_size:point(1)+Gap_size,point(2)-Gap_size:point(2)+Gap_size)==1);
    end

    % Extract edge towards another direction. Find curvature local maxima as corner candidates. 
    % The 'min' only get one minimum value every time.
    point=[r(1),c(1)];
    BW1(point(1),point(2))=0;
    [I,J]=find(BW1(point(1)-Gap_size:point(1)+Gap_size,point(2)-Gap_size:point(2)+Gap_size)==1);
    while size(I,1)>0
        dist=(I-Gap_size-1).^2+(J-Gap_size-1).^2;
        [min_dist, index]=min(dist);
        point=point+[I(index),J(index)]-Gap_size-1;
        cur=[point;cur];
        BW1(point(1),point(2))=0;
        [I,J]=find(BW1(point(1)-Gap_size:point(1)+Gap_size,point(2)-Gap_size:point(2)+Gap_size)==1);
    end

    if size(cur,1)>(size(BW,1)+size(BW,2))/25   % if the size of the curve of larger than 1/25 of 
                                                % the original image, the curve is kept, otherwise is discarded

        cur_num=cur_num+1;
        curve{cur_num}=cur-Gap_size;
    end
    [r,c]=find(BW1==1);

end
curve_start = [];
curve_end = [];
curve_mode = [];
for i=1:cur_num
    curve_start(i,:)=curve{i}(1,:);
    curve_end(i,:)=curve{i}(end,:);
    if (curve_start(i,1)-curve_end(i,1))^2+...
            (curve_start(i,2)-curve_end(i,2))^2<=32
        curve_mode(i,:) = 'loop';
    else
        curve_mode(i,:) = 'line';
    end
    curve{i} = connect_loop(curve{i}(:, :), curve_mode(i,:));
    curve_start(i,:)=curve{i}(1,:);
    curve_end(i,:)=curve{i}(end,:);
    BW_edge(curve{i}(:,1)+(curve{i}(:,2)-1)*L)=1;
end
[curve, BW_edge, connect_state, T_point_state, cur_num] = ...;
     connect_line_and_T(curve,curve_start,curve_end,curve_mode,cur_num,BW_edge);

% -------------------------------------------------------------------------
function xy_cord = connect_loop(xy_cord, corner_model)
% This function is used to fill the two end points on the same contour when
% the distance between them is not more than a fixed size and transform the
% open contour to a loop one.
% -------------------------------------------------------------------------
if corner_model == 'loop'
    p1 = xy_cord(1, :);
    p2 = xy_cord(end, :);
    if abs(p1(1) - p2(1)) > abs(p1(2) - p2(2))
        step = (p2(1) - p1(1))/abs(p2(1) - p1(1));
        k = (p1(2) - p2(2))/(p1(1) - p2(1));
        for i = p1(1)+step : step : p2(1)-step
            x = i;
            y = round(p1(2) + k*(i - p1(1)));
            xy_cord = [[x,y]; xy_cord];
        end
    else
        step = (p2(2) - p1(2))/abs(p2(2) - p1(2));
        k = (p1(1) - p2(1))/(p1(2) - p2(2));
        for i = p1(2)+step : step : p2(2)-step
            y = i;
            x = round(p1(1) + k*(i - p1(2)));
            xy_cord = [[x,y]; xy_cord];
        end
    end
end

% -------------------------------------------------------------------------
function [curve, BW_edge, connect_state, T_point_state, cur_num] = ...;
connect_line_and_T(curve, curve_start, curve_end, curve_mode, cur_num, BW_edge)
% This function is used to stretch the end of the open contour to the nearest 
% edge pixel on the other contour when the neighborhood of the end of one open 
% contour contains edge pixel on other contour.
% ------------------------------------------------------------------------- 
% Input:
%   (1)curve           ---The extracted curves whose each element is an individual chain 
%                         code corresponding a contour.
%   (2)curve_start     --- The set of the first pixel's location of each contour. 
%   (3)curve_end       --- The set of the last pixel's location of each contour.
%   (4)curve_mode      --- The set of the the type of each contour.It's either
%                          'loop' for a loop contour or 'line' for an open contour.
%   (5)cur_num         --- The number of the contours.
%   (6)BW_edge         --- The edge map just after filling the small gap.
% Output:
%   (1)curve           ---The extracted curves whose each element is an individual chain 
%                         code corresponding a contour.
%   (2)BW_edge         --- The edge map after Stretch the open contours.
%   (3)connect_state   --- The state of all the end points.If connect_state(i, 1) = 1,
%                          it means that the start point of i-th contour remains unchanged.
%                          If connect_state(i, 2) = 1,it means that the last point of i-th 
%                          contour remains unchanged.
%   (4)T_point_state   --- The T-type state of all the end points.If T_point_state(i, 1) = 1,
%                          it means that the start point of i-th contour is a T-type corner.
%                          If connect_state(i, 2) = 1,it means that the last point of i-th 
%                          contour is a T-type corner.
%   (5)cur_num         --- The number of the contours.
% -------------------------------------------------------------------------
dis = 4;
% num = cur_num;
[rows, cols] = size(BW_edge);
im_end = zeros(size(BW_edge));
connect_state = zeros(cur_num, 2);  
T_point_state = connect_state;      

jth_line_connected = [];
for i = 1 : cur_num
%     curve_mode(i, :)
%     strcmp(curve_mode(i, :), 'line')
    if sum('line'== curve_mode(i, :)) == 4
        connect_state(i, :) = [1, 1];
        im_end(curve_start(i, 1), curve_start(i, 2)) = 1;
        im_end(curve_end(i, 1), curve_end(i, 2)) = 1;
    end
end

i = 1;
while i < cur_num
    ds = [];
    if connect_state(i, 1) == 1
        p1 = curve_start(i, :);
        for j = i+1 : cur_num
            if connect_state(j, 1) == 1
                p2 = curve_start(j, :);
                d_p1_p2 = sum((p1-p2).^2);
                ds = [ds; [j, 1, d_p1_p2]];  % the jth curve, start1/end2, distance between p1 and p2
            end
            if connect_state(j, 2) == 1
                p2 = curve_end(j, :);
                d_p1_p2 = sum((p1-p2).^2);
                ds = [ds; [j, 2, d_p1_p2]];
            end
        end
    end
    if ~isempty(ds) 
        % Among the candidate end points which are within i-th contour's end point's neighborhood,
        % search the nearest one. 
        [min_dis, index] = min(ds(:, 3));
        if ~isempty(min_dis) && min_dis < 2*dis^2+1
            i_ind = i;
            j_ind = ds(index, 1);
            start_or_end = ds(index, 2);
            connect_state(i_ind, 1) = 0;
            connect_state(j_ind, start_or_end) = 0;
            p1 = curve_start(i_ind, :);
            if start_or_end == 1
                p2 = curve_start(j_ind, :);
                connect_state(j_ind, 1) = 0;
            else
                p2 = curve_end(j_ind, :);
                connect_state(j_ind, 2) = 0;
            end
            connect_state(i_ind, 1) = 0;
            xy_cord = insert_point(p2, p1);
            curve{i} = [xy_cord; curve{i}];
        end
    end
    
    ds = [];
    if connect_state(i, 2) == 1
        p1 = curve_end(i, :);
        for j = i+1 : cur_num
            if connect_state(j, 1) == 1
                p2 = curve_start(j, :);
                d_p1_p2 = sum((p1-p2).^2);
                ds = [ds; [j, 1, d_p1_p2]];  % the jth curve, start=1/end=2, distance between p1 and p2
            end
            if connect_state(j, 2) == 1
                p2 = curve_end(j, :);
                d_p1_p2 = sum((p1-p2).^2);
                ds = [ds; [j, 2, d_p1_p2]];
            end
        end
    end
    if ~isempty(ds)
        [min_dis, index] = min(ds(:, 3));
        if ~isempty(min_dis) && min_dis < 2*dis^2+1
            i_ind = i;
            j_ind = ds(index, 1);
            start_or_end = ds(index, 2);
            connect_state(i_ind, 1) = 0;
            connect_state(j_ind, start_or_end) = 0;
            p1 = curve_end(i_ind, :);
            if start_or_end == 1
                p2 = curve_start(j_ind, :);
                connect_state(j_ind, 1) = 0;
            else
                p2 = curve_end(j_ind, :);
                connect_state(j_ind, 2) = 0;
            end
            connect_state(i_ind, 2) = 0;
            xy_cord = insert_point(p1, p2);
            curve{i} = [curve{i}; xy_cord];
        end
    end
    i = i + 1;
end
for i=1:cur_num
    curve_start(i,:)=curve{i}(1,:);
    curve_end(i,:)=curve{i}(end,:);
    BW_edge(curve{i}(:,1)+(curve{i}(:,2)-1)*rows) = 1;
end
im_end = zeros(size(im_end));
for i = 1 : cur_num

    if sum('line'== curve_mode(i, :)) == 4
        im_end(curve_start(i, 1), curve_start(i, 2)) = 1;
        im_end(curve_end(i, 1), curve_end(i, 2)) = 1;
    end
end

im_tmp = zeros(size(im_end)+2*dis);
im_tmp(dis+1:dis+rows, dis+1:dis+cols) = BW_edge;
points_num = 10; % point num that need to be removed

for i = 1 : cur_num-1
    if connect_state(i, 1) == 1
        im_tmp(curve{i}(1:points_num, 1)+dis +(dis+(curve{i}(1:points_num, 2) - 1))*(rows+2*dis)) = 0;
        point = curve_start(i,:)+dis;
        [I, J] = find(im_tmp(point(1)-dis:point(1)+dis, point(2)-dis:point(2)+dis) == 1);
        im_tmp(curve{i}(1:points_num, 1)+dis + (dis+(curve{i}(1:points_num, 2) - 1))*(rows+2*dis)) = 1;
        
        if length(I) > 0
            dist=(I-dis-1).^2+(J-dis-1).^2;
            [min_dist, index] = min(dist);
            p1 = point;
            p2 = p1+[I(index),J(index)]-dis-1;
            xy_cord = insert_point(p1, p2)-dis;
            curve{i} = [xy_cord(end:-1:1, :); curve{i}];
            connect_state(i, 1) = 0;
            T_point_state(i, 1) = 1;
        end  
    end
    if connect_state(i, 2) == 1
        im_tmp(curve{i}(end-points_num:end, 1)+dis +(dis+(curve{i}(end-points_num:end, 2) - 1))*(rows+2*dis)) = 0;
        point = curve_end(i,:)+dis;
        [I, J] = find(im_tmp(point(1)-dis:point(1)+dis, point(2)-dis:point(2)+dis) == 1);
        im_tmp(curve{i}(1:points_num, 1)+dis + (dis+(curve{i}(1:points_num, 2) - 1))*(rows+2*dis)) = 1;
        % if there is at least one edge point around the original point, go on;
        if length(I) > 0
            dist=(I-dis-1).^2+(J-dis-1).^2;
            [min_dist, index] = min(dist);
            p1 = point;
            p2 = p1+[I(index),J(index)]-dis-1;
            xy_cord = insert_point(p1, p2)-dis;
            curve{i} = [curve{i}; xy_cord];
            connect_state(i, 2) = 0;
            T_point_state(i, 2) = 1;
        end
    end
end


for i=1:cur_num
    curve_start(i,:)=curve{i}(1,:);
    curve_end(i,:)=curve{i}(end,:);
    BW_edge(curve{i}(:,1)+(curve{i}(:,2)-1)*rows) = 1;
end
im_end = zeros(size(im_end));
for i = 1 : cur_num

    if sum('line'== curve_mode(i, :)) == 4
        im_end(curve_start(i, 1), curve_start(i, 2)) = 1;
        im_end(curve_end(i, 1), curve_end(i, 2)) = 1;
    end
end

% -------------------------------------------------------------------------
function xy_cord = insert_point(p1, p2)
% This function is used to link two points and fill the gap between them.
% -------------------------------------------------------------------------
% Input:
%   (1)p1        --- The first point to be linked.
%   (2)p2        --- The second point to be linked.
% Output:
%   xy_cord      --- The chain code which consists of the two points and the 
%                    points between them. 
% -------------------------------------------------------------------------
xy_cord = [];
if abs(p1(1) - p2(1)) > abs(p1(2) - p2(2))
    step = (p2(1) - p1(1))/abs(p2(1) - p1(1)); % 1 or -1
    k = (p1(2) - p2(2))/(p1(1) - p2(1));
    for i = p1(1)+step : step : p2(1)-step
        x = i;
        y = round(p1(2) + k*(i - p1(1)));
        xy_cord = [xy_cord; [x,y]];
    end
else
    step = (p2(2) - p1(2))/abs(p2(2) - p1(2));
    k = (p1(1) - p2(1))/(p1(2) - p2(2));
    for i = p1(2)+step : step : p2(2)-step
        y = i;
        x = round(p1(1) + k*(i - p1(2)));
        xy_cord = [xy_cord; [x,y]];
    end
end

% =========================================================================
function myANDDofUCM=ANDDofUCM(sigma2,rho2,stren_ve,angle_ve,p)
% This function is used to obtain the ANDD representation of a universal corner model
% with a set of arbitrary number BCFs of specified angle ¦Âs and basic component strength Ts.
% -------------------------------------------------------------------------
% Input:
%   (1)sigma2          --- The square of the scale factor
%   (2)rho2            --- The square of the anisotropic scale factor
%   (3)stren_ve        --- The basic component strength Ts vector,each element in vector indicate a
%                          basic component strength.
%   (4)angle_ve        --- The angle ¦Âs vector and it has the same number
%                          element as the stren_ve.
%   (5)p               --- The orientation number 
% Output:
%   myANDDofUCM        --- The ANDD representation of a universal corner model(UCM)
% -------------------------------------------------------------------------
rho = sqrt(rho2);
sigma = sqrt(sigma2);
theta=0:2*pi/p:2*pi-2*pi/p;
myANDDofUCM=zeros(1,length(theta));
len=length(stren_ve);
if length(stren_ve) ~= length(angle_ve)
    eid=sprintf('Images:%s:thelengthofstrengthandanglemustbesame',mfilename);
    msg='the length of strength and angle must be same';
    error(eid,'%s',msg);
end
stren_ve2=[stren_ve(len) stren_ve];
angle_ve2=angle_ve;

for i=1:len
    BCF=[];
    BCF=rho/(sqrt(8*pi)*sigma)*(stren_ve2(i+1)-stren_ve2(i))*cos(theta-angle_ve2(i))./...;
        sqrt(cos(theta-angle_ve2(i)).*cos(theta-angle_ve2(i))+rho^4*sin(theta-angle_ve2(i)).*sin(theta-angle_ve2(i)));
    myANDDofUCM=myANDDofUCM+BCF;
end

% =========================================================================
function [r,c] = nonmaxsuppts(cim, radius,thresh)
% Extract local maxima by performing a grey scale morphological
% dilation and then finding points in the corner strength image that
% match the dilated image and are also greater than the threshold.
% -------------------------------------------------------------------------
% Input:
%   (1)cim       --- The corner measure matrix
%   (2)radius    --- The window width of Non-maximum suppression is 2*radius+1.
%   (3)thresh    --- The thresh of Thresholding.
% Output:
%   (1)r         --- The detected corners' row index.
%   (2)c         --- The detected corners' column index.
% -------------------------------------------------------------------------

sze = 2*radius+1;                   % Size of dilation mask.
mx = ordfilt2(cim,sze^2,ones(sze)); % Grey-scale dilate.

% Make mask to exclude points within radius of the image boundary.
bordermask = zeros(size(cim));
bordermask(radius+1:end-radius, radius+1:end-radius) = 1;

% Find maxima, threshold, and apply bordermask
cimmx = (cim==mx) & cim>thresh & bordermask;
[r,c] = find(cimmx);                % Find row,col coords.

% =========================================================================
function classification = corner_classify(corner_index,extend_image,ANDD_FilterBank,tao)
% This function is used to classify the detected corner through the peak number of each
% corner's ANDD represent.
% -------------------------------------------------------------------------
% Input:
%   (1)corner_index      -- The location of detected corner,each row for a corner.
%   (2)extend_image      -- The extended one of the input image.
%   (3)ANDD_FilterBank   -- The generated anisotropic Directional derivative filters.
%   (4)tao               -- The threshold ¦Ó.
% Output:
%   (5)classification    -- The corner classification result.In each row, the first 
%                           two values indicate a detected corner location and the last
%                           value is the corner type flag corresponding the corner.
% -------------------------------------------------------------------------
classification = [];
width = floor(size(ANDD_FilterBank,1)/2);
P     = size(ANDD_FilterBank,3);

for i = 1:size(corner_index,1)
    template =[];
    peak_num = 0;
     for direction=1:floor(P/2)
           template(direction) =sum(sum(extend_image(corner_index(i,1)-width:corner_index(i,1)+width,...;
               corner_index(i,2)-width:corner_index(i,2)+width).*ANDD_FilterBank(:,:,direction))); 
     end
        % Normalising the magnitudes
        template = abs(template/max(abs(template)));
        template = [template(end);template(:);template(1)];
        for j = 2:length(template)-1
            if template(j)>template(j-1) & template(j)>template(j+1) &template(j)>tao
                peak_num = peak_num+1;
            end
        end
       classification = [classification; corner_index(i,:)-width peak_num]; 
end

% =========================================================================
function [im,eta,rou2,canny_thresh,sigma,isClassifing]=parse_inputs(varargin)
% This function is used to initialize the parameter.
% -------------------------------------------------------------------------
error(nargchk(0,6,nargin));
Para={0.12,6,[0.1,0.35],1.0,1}; %Default experience value;

if nargin>= 2
    im=varargin{1};
    for i = 2:nargin
        if size(varargin{i},1)>0
            Para{i-1} = varargin{i};
        end
    end
end
if nargin == 1
    im=varargin{1};
end   
if nargin == 0 | size(im,1) == 0
    [fname,dire] = uigetfile('*.bmp;*.jpg;*.gif','Open the image to be detected');
    im = imread([dire,fname]);
end

eta             = Para{1};
rou2            = Para{2};
canny_thresh    = Para{3};
sigma           = Para{4};
isClassifing    = Para{5};