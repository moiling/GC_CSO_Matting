%
% 每个区域选一个点，将大图缩小的抠图方式
%
clear;
addpath(genpath(pwd));

max_fitness_evaluation = 1e6;

path = '../data/train/';
img_path = [path, 'input/input_training_highres/'];
trimap_path = [path, 'trimap/trimap_training_highres/Trimap1/'];
gt_path = [path, 'ground_truth/gt_training_highres/'];
img_dir = dir([img_path, '*.png']);
cut_size = 20;

for ii = 1

    img_url = [img_path, img_dir(ii).name];
	trimap_url = [trimap_path, img_dir(ii).name];
	img = imread(img_url);
	trimap = imread(trimap_url);
      
    [H, W, C] = size(img);
    
    img_rgb = single(reshape(img, W*H, C));
    trimap_alpha = single(reshape(trimap, W*H, 1));
    
    %% small cut function
    for jj = 1:cut_size * cut_size     
        mask = zeros(cut_size * cut_size, 1, 'uint8'); % [0, 0, 0, 0]
        mask(jj) = 1; % [1, 0, 0, 0]
        mask = reshape(mask, cut_size, cut_size);
        mask = repmat(mask, ceil(H / cut_size), ceil(W / cut_size));
        mask = mask(1:H, 1:W);
        
        s_H = 0;
        for kk = 1:cut_size
            s_H = max(s_H, length(find(mask(:, kk) == 1)));
        end
        s_W = 0;
        for kk = 1:cut_size
            s_W = max(s_W, length(find(mask(kk, :) == 1)));
        end

        mask_ind = find(mask == 1);
        small_img = img_rgb(mask_ind, :);
        small_img = uint8(reshape(small_img, s_H, s_W, C));
        small_trimap = trimap_alpha(mask_ind, :);
        small_trimap = uint8(reshape(small_trimap, s_H, s_W, 1));
        
        for nn = 1:100
        % CSO
        [alpha_matte, fitness, x] = CSO(small_img, small_trimap, max_fitness_evaluation, small_trimap, nn);
        
        file_name = sprintf('%d-%d-%s-%d-%d', nn, max_fitness_evaluation, img_dir(ii).name(1:end-4), cut_size, jj);
        imwrite(alpha_matte, ['./result/cso-small2/', file_name, '.png']);
        save(['./result/cso-small2/', file_name, '.mat'], 'x', 'fitness');
        
        % F,B
        F_ind = find(small_trimap == 255); 
        B_ind = find(small_trimap == 0); 
        U_ind = find(small_trimap == 128); 
        img_rgb = single(reshape(small_img, [s_H * s_W, 3]));
        F_rgb = img_rgb(F_ind, :);
        B_rgb = img_rgb(B_ind, :);
        U_rgb = img_rgb(U_ind, :);
        
        F = reshape(small_img, s_W * s_H, 3);
        B = reshape(small_img, s_W * s_H, 3);
        x = round(x);
        for kk = 1:length(U_ind)
            F(U_ind(kk), :) = F_rgb(x(kk), :);
            B(U_ind(kk), :) = B_rgb(x(kk + length(U_ind)), :);
        end
        F = reshape(F, s_H, s_W, 3);
        B = reshape(B, s_H, s_W, 3);

        F_trimap = small_trimap;
        B_trimap = small_trimap;
        F_trimap(F_trimap == 128) = 255;
        B_trimap(B_trimap == 128) = 0;
        B_trimap = 255 - B_trimap;
        imwrite(F, ['./result/cso-small2/F/', file_name, '.png']);
        imwrite(B, ['./result/cso-small2/B/', file_name, '.png']);
        imwrite(F, ['./result/cso-small2/cut/', file_name, '.png'], 'Alpha', double(alpha_matte) / 255);

        FF_BB_sample = reshape(small_img, s_H * s_W, 3);
        FF_BB_sample(F_ind(x(1:length(U_ind))), :) = repmat([255, 255, 255], length(U_ind), 1);
        FF_BB_sample(B_ind(x(length(U_ind) + 1:end)), :) = repmat([0, 0, 0], length(U_ind), 1);
        FF_BB_sample = reshape(FF_BB_sample, s_H, s_W, 3);
        imwrite(FF_BB_sample, ['./result/cso-small2/F-B/', file_name, '.png']);
        end
    end
end
