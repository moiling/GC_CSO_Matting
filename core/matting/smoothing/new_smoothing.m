%
% 平滑处理操作，confidence使用的是global matting论文中的评价方式
%
clear;
warning('off');

data_path = './data/train/';
out_path = './result/eval-5.000000e+03-mask-11/';
mask_path = [out_path, 'mask/'];
img_path = [data_path, 'input/input_training_highres/'];
trimap_path = [data_path, 'trimap/trimap_training_highres/Trimap1/'];
gt_path = [data_path, 'ground_truth/gt_training_highres/'];
img_dir = dir([img_path, '*.png']);
sub_dir = 'cc-pso/';

for m = 1:27
    for ii = 1:10
        disp(img_dir(m).name);
        img_url = [img_path, img_dir(m).name];
        trimap_url = [mask_path, img_dir(m).name];
        file_name = sprintf('GT%02d_iter_%d_without_smoothing.png', m, ii);        
        alpha_url = [out_path, sub_dir, file_name];

        img = imread(img_url);
        [W, H, ~] = size(img);
        trimap = imread(trimap_url);
        alpha_matte = imread(alpha_url);

        U_ind = find(trimap == 128);
        F_ind = find(trimap == 255);
        B_ind = find(trimap == 0);

        img_rgb = single(reshape(img,[numel(trimap), 3]));

        F_rgb = img_rgb(F_ind,:);
        B_rgb = img_rgb(B_ind,:);
        U_rgb = img_rgb(U_ind,:);
        pic_name = img_dir(m).name(1:end-4);

        mat_file_name = sprintf('GT%02d_iter_%d.mat', m, ii);
        load([out_path, sub_dir, mat_file_name]);

        U_alpha = reshape(alpha_matte, W * H, 1);
        U_alpha = single(U_alpha(U_ind)) / 255;
        
        % 计算confidence
        fitness = ones(length(U_ind), 1);   
        x = round(x);
        for jj = 1:length(U_ind)
            fitness(jj) = norm2(U_rgb(jj) - (U_alpha(jj) .* F_rgb(x(jj)) + (1 - U_alpha(jj)) .* B_rgb(x(jj + length(U_ind)))));
        end

        % Post-processing which is mentioned in paper titled "A Global Sampling Method for Alpha Matting"
        trimap = trimap(:, :, 1);
        alpha_matte = alpha_matte(:, :, 1);

        trimap = reshape(trimap, W, H, 1);
        img = reshape(img, W, H, 3);
        img = double(img);
        confidence = ones(W * H, 1);

        confidence(U_ind) = exp(-fitness / 2);
        confidence = reshape(confidence, W, H, 1);
        confidence = confidence .* 255;

        alpha_matte = smoothing(img, alpha_matte, confidence, trimap);

        imwrite(alpha_matte, [out_path, sub_dir, 'smoothing/', file_name]);
    end
end