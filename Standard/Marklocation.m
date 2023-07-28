function img = Marklocation(im,locx,locy,markwidth)
    if 3 == size(im,3)
        im = rgb2gray(im); % Transform RGB image to a Gray one.
    end

    [rows cols] = size(im);
    wid = 7;
    %%%%%%%%% Extend the image %%%%%%%%%
    extend_image = zeros(rows+2*wid,cols+2*wid);

    extend_image(wid+1:rows+wid,wid+1:cols+wid) = im;

    for i = 1:size(extend_image,1)                         
        if i<=wid
            r = 2*wid+1-i;
        elseif i>rows + wid
            r = 2*(rows+wid)+1-i;
        else
            r = i;
        end
        for j = 1:size(extend_image,2)
            if j<=wid
                c = 2*wid+1-j;
            elseif j>cols + wid
                c = 2*(cols+wid)+1-j;
            else
                c = j;
            end
            extend_image(i,j) = extend_image(r,c);
        end
    end
    
    ssss = extend_image(wid+1:rows+wid,wid+1:cols+wid);
    for i=1:size(locx(:,1))

            ssss = mark(ssss,locx(i),locy(i),markwidth);

    end
    img = [];
    figure;
    imshow(ssss,[]);
       
end


function img1=mark(img,x,y,w)

    x = round(x);
    y = round(y);

    [M,N,C]=size(img);
    img1=img;

    if isa(img,'logical')
        img1(max(1,x-floor(w/2)):min(M,x+floor(w/2)),max(1,y-floor(w/2)):min(N,y+floor(w/2)),:) = ...
            (img1(max(1,x-floor(w/2)):min(M,x+floor(w/2)),max(1,y-floor(w/2)):min(N,y+floor(w/2)),:)<1);
        img1(x-floor(w/2)+1:x+floor(w/2)-1,y-floor(w/2)+1:y+floor(w/2)-1,:)=...
            img(x-floor(w/2)+1:x+floor(w/2)-1,y-floor(w/2)+1:y+floor(w/2)-1,:);
    else
        img1(max(1,x-floor(w/2)):min(M,x+floor(w/2)),max(1,y-floor(w/2)):min(N,y+floor(w/2)),:) = ...
            (img1(max(1,x-floor(w/2)):min(M,x+floor(w/2)),max(1,y-floor(w/2)):min(N,y+floor(w/2)),:)<128)*255;
        img1(max(1,x+1-floor(w/2)):min(M,x-1+floor(w/2)),max(1,y+1-floor(w/2)):min(N,y-1+floor(w/2)),:) = ...
            img(max(1,x+1-floor(w/2)):min(M,x-1+floor(w/2)),max(1,y+1-floor(w/2)):min(N,y-1+floor(w/2)),:);

    end
    
end
