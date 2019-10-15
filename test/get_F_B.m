%
% 显示采样的颜色和点的位置
%
clear;
warning('off');

data_path = '../data/train/';
out_path = '../result/eval-5.000000e+03-mask-11/';
mask_path = [out_path, 'mask/'];
img_path = [data_path, 'input/input_training_highres/'];
trimap_path = [data_path, 'trimap/trimap_training_highres/Trimap1/'];
gt_path = [data_path, 'ground_truth/gt_training_highres/'];
img_dir = dir([img_path, '*.png']);
sub_dir = 'no-cc-cso2/';

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


for m = 5
    tic;
    disp(img_dir(m).name);
    img_url = [img_path, img_dir(m).name];
    trimap_url = [mask_path, img_dir(m).name];
    file_name = sprintf('GT%02d_iter_1_without_smoothing.png', m);
    
    alpha_url = [out_path, sub_dir, file_name];
    
    img = imread(img_url);
    [W, H, ~] = size(img);
    trimap = imread(trimap_url);
    alpha_matte = imread(alpha_url);

    U_ind = find(trimap == 128);
    F_ind = find(trimap == 255);
    B_ind = find(trimap == 0);
    
    img_rgb = single(reshape(img,[numel(trimap),3]));
    
    F_rgb = img_rgb(F_ind,:);
    B_rgb = img_rgb(B_ind,:);
    U_rgb = img_rgb(U_ind,:);
    F_s = ind2sub(size(trimap), F_ind);
    B_s = ind2sub(size(trimap), B_ind);
    U_s = ind2sub(size(trimap), U_ind);
    pic_name = img_dir(m).name(1:end-4);
    
    mat_file_name = sprintf('GT%02d_iter_1.mat', m);
    load([out_path, sub_dir, mat_file_name]);
    
    U_alpha = reshape(alpha_matte, W * H, 1);
    U_alpha = single(U_alpha(U_ind)) / 255;
    % 计算confidence
    
    F = reshape(img, W*H, 3);
    B = reshape(img, W*H, 3);
    x = round(x);
    for ii = 1:length(U_ind)
        F(U_ind(ii), :) = F_rgb(x(ii), :);
        B(U_ind(ii), :) = B_rgb(x(ii+length(U_ind)), :);
    end
    F = reshape(F, W, H, 3);
    B = reshape(B, W, H, 3);
    
    toc;
    F_trimap = trimap;
    B_trimap = trimap;
    F_trimap(F_trimap == 128) = 255;
    B_trimap(B_trimap == 128) = 0;
    B_trimap = 255 - B_trimap;
    imwrite(F, [out_path, sub_dir, 'F_sample/', file_name]);
    imwrite(B, [out_path, sub_dir, 'B_sample/', file_name]);
    imwrite(F, [out_path, sub_dir, 'F/', file_name], 'Alpha', double(alpha_matte)/255);
    
    FF_BB_sample = reshape(img, W*H,3);
    FF_BB_sample(F_ind(x(1:length(U_ind))), :) = repmat([255,255,255], length(U_ind), 1);
    FF_BB_sample(B_ind(x(length(U_ind)+1:end)), :) = repmat([0,0,0], length(U_ind), 1);
    FF_BB_sample = reshape(FF_BB_sample, W,H,3);
    imwrite(FF_BB_sample, [out_path, sub_dir, 'F_B_sample/', file_name]);
end