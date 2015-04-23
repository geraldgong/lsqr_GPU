clear all; close all; clc

k = 10; % step times
n = 31;
nz = 65;
posx = [1:n]; posy = [1:n];
[Posy Posx] = meshgrid(posx,posy);
Posx_vec = uint32(Posx(:));
Posy_vec = uint32(Posy(:));
load('A_mat_31_25pt.mat');
A = single(full(A_mat));
b = single(b_vec);

% Tikhonov Regularization
lambda = 0.5;
% I = eye(size(A,2));
% A = [A;lambda^2 * I];
% b = reshape(b, [size(A_mat,1), n*n]);
% b = [b; zeros(size(A,2),n*n)];
% b = b(:);

tic;
M = lsqr_sv_Cuda(A, b, Posx_vec, Posy_vec, n, nz, k, lambda);
toc;

X = M(:,end);
image_final =  reshape(X,[n, n, nz]);

% representing the image

% top view
figure(); imagesc(squeeze(max(image_final,[],3)));
colormap(hot);
% lateral view
figure(); imagesc(squeeze(max(image_final,[],2)));
colormap(hot);