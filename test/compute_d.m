%
% 计算维度
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

d = zeros(27, 2);
for ii = 1:length(img_dir)
    disp(ii);
	% 读资源
	img_url = [img_path, img_dir(ii).name];
	trimap_url = [trimap_path, img_dir(ii).name];
	img = imread(img_url);
	trimap = imread(trimap_url);
	  
	% 统一做预处理
	mask = imread([mask_path, img_dir(ii).name]);

    % 预处理之后三分图的区域
    F_ind = find(mask == 255); 
    B_ind = find(mask == 0); 
    U_ind = find(mask == 128); 

    % 读取三个区域的颜色数据
    img_rgb = single(reshape(img, [numel(trimap), 3]));
    F_rgb = img_rgb(F_ind, :);
    B_rgb = img_rgb(B_ind, :);
    U_rgb = img_rgb(U_ind, :);

    [cc_rgb, U_cc_ind, cc_U_ind] = color_clustering(U_rgb);
    d(ii, 1) = length(U_ind);
    d(ii, 2) = length(cc_U_ind);
end