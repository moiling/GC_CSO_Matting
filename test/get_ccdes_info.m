%
% 显示CC-DE-S采样的颜色和点的位置
%
clear;
warning('off');
addpath(genpath(pwd));

data_path = '../data/train/';
out_path = '~/Downloads/low/';
mask_path = ['../result/eval-5.000000e+03-mask-11/', 'mask/'];
img_path = [data_path, 'input/input_training_lowres/'];
trimap_path = [data_path, 'trimap/trimap_training_lowres/Trimap1/'];
gt_path = [data_path, 'ground_truth/gt_training_lowres/'];
img_dir = dir([img_path, '*.png']);
sub_dir = '';

if ~exist([out_path, sub_dir, 'F_sample'], 'dir')
    mkdir([out_path, sub_dir, 'F_sample']);
end
if ~exist([out_path, sub_dir, 'B_sample'], 'dir')
    mkdir([out_path, sub_dir, 'B_sample']);
end
if ~exist([out_path, sub_dir, 'F'], 'dir')
    mkdir([out_path, sub_dir, 'F']);
end
if ~exist([out_path, sub_dir, 'F_B_sample'], 'dir')
    mkdir([out_path, sub_dir, 'F_B_sample']);
end

for jj = 1:1992
for m = 5
    disp(img_dir(m).name);
    img_url = [img_path, img_dir(m).name];
    trimap_url = [trimap_path, img_dir(m).name];
        
    file_name = sprintf('%d_%d_no_smoothing.png', m, jj * 10);
    
    alpha_url = [out_path, sub_dir, file_name];  
    
    img = imread(img_url);
    [W, H, ~] = size(img);
    trimap = imread(trimap_url);
    alpha_matte = imread(alpha_url);
    trimap = TrimapExpansion(double(img), trimap, 11);
    
    U_ind = find(trimap == 128);
    F_ind = find(trimap == 255);
    B_ind = find(trimap == 0);
    
    img_rgb = single(reshape(img, [numel(trimap), 3]));
    
    F_rgb = img_rgb(F_ind, :);
    B_rgb = img_rgb(B_ind, :);
    U_rgb = img_rgb(U_ind, :);
    F_s = ind2sub(size(trimap), F_ind);
    B_s = ind2sub(size(trimap), B_ind);
    U_s = ind2sub(size(trimap), U_ind);
    pic_name = img_dir(m).name(1:end-4);
    
    mat_file_name = sprintf('%d_%d.mat', m, jj * 10);
    load([out_path, sub_dir, mat_file_name]);
    load([out_path, 'SetBSample.mat']);
    load([out_path, 'SetFSample.mat']);
    load([out_path, 'SetUn.mat']);
    load([out_path, 'SetUnNum.mat']);
    x = gbest;
    U_alpha = reshape(alpha_matte, W * H, 1);
    U_alpha = single(U_alpha(U_ind)) / 255;

    F = img;
    B = img;
    x = round(x);

    FF_BB_sample = img;
    for ii = 1:length(U_ind)
        F(Un(ii, 1), Un(ii, 2), :) = FSample(x(ii * 2 - 1), 1:3);
        B(Un(ii, 1), Un(ii, 2), :) = BSample(x(ii * 2), 1:3);
        
        FF_BB_sample(FSample(x(ii * 2 - 1), 4), FSample(x(ii * 2 - 1), 5), :) = [255, 255, 255];
        FF_BB_sample(BSample(x(ii * 2), 4), BSample(x(ii * 2), 5), :) = [0, 0, 0];
    end
    
    F_trimap = trimap;
    B_trimap = trimap;
    F_trimap(F_trimap == 128) = 255;
    B_trimap(B_trimap == 128) = 0;
    B_trimap = 255 - B_trimap;
    imwrite(F, [out_path, sub_dir, 'F_sample/', file_name]);
    imwrite(B, [out_path, sub_dir, 'B_sample/', file_name]);
    imwrite(F, [out_path, sub_dir, 'F/', file_name], 'Alpha', double(alpha_matte) / 255);
    
    imwrite(FF_BB_sample, [out_path, sub_dir, 'F_B_sample/', file_name]);
end
end