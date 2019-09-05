%
% 计算MSE
%
clear;
warning('off');

data_path = './data/train/';
out_path = './result/eval-5.000000e+03-mask-11/';
img_path = [data_path, 'input/input_training_highres/'];
trimap_path = [data_path, 'trimap/trimap_training_highres/Trimap1/'];
gt_path = [data_path, 'ground_truth/gt_training_highres/'];
img_dir = dir([img_path, '*.png']);
sub_dir = 'cc-de-s/smoothing/';

for ii = 1
error = zeros(27, 1);
    for m = 12
        disp(img_dir(m).name);
        img_url = [img_path, img_dir(m).name];
        trimap_url = [trimap_path, img_dir(m).name];
        %alpha_url = [out_path, img_dir(m).name];
        %trimap_url = './result/high-cc-pre-5e3/GT04_mask_13.png';

        file_name = sprintf('GT%02d_iter_1_without_smoothing.png', m);
        alpha_url = [out_path, sub_dir, file_name];

        img = imread(img_url);
        trimap = imread(trimap_url);
        alpha_matte = imread(alpha_url);

        U_ind = find(trimap == 128);

        alpha_matte = im2single(alpha_matte);

        gt = im2single(imread([gt_path,img_dir(m).name]));
        U_gt = gt(U_ind);
        pso_alpha_error = sum(abs(alpha_matte(U_ind)-U_gt).^2) / length(U_ind);
        fprintf('%d\n', pso_alpha_error);
        error(m) = pso_alpha_error;
    end
    error_file_name = sprintf('%d_error.mat', ii);
    save([out_path, sub_dir, error_file_name], 'error');
end