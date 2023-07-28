% load('standard_geometric.mat')
% StandardCorner=geometric;
% I=imread('ss.png');

load('standardlab.mat')
StandardCorner=lab;
I=imread('l.gif');

[lx,ly]=find(StandardCorner);
% imshow(I);
% hold on;
% plot(ly,lx,'r.');
img = Marklocation(I,lx,ly,5);