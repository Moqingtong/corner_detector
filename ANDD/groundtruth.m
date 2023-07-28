clc
clear all
close all

load('standardlab.mat')
Standard_Corner=lab;
I=imread('l.gif');
% load('standard_geometric.mat')
% Standard_Corner=geometric;
% I=imread('ss.png');

[~,detect_Corner] = cornerXC(I);
[missed_corner, false_corner, local,detected_num,standard_num] = match_corner(detect_Corner,Standard_Corner);
[lx,ly]=find(detect_Corner);
% imshow(I);
% hold on;
% plot(ly,lx,'c*');
img = Marklocation(I,lx,ly,5);