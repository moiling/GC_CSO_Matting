%
% 每个区域选一个点，将大图缩小的抠图方式
% 用暴力穷举法观察结果
%
clear;
addpath(genpath(pwd));

path = '../data/train/';
img_path = [path, 'input/input_training_highres/'];
trimap_path = [path, 'trimap/trimap_training_highres/Trimap1/'];
gt_path = [path, 'ground_truth/gt_training_highres/'];
img_dir = dir([img_path, '*.png']);
cut_size = 20;

for ii = 5

    img_url = [img_path, img_dir(ii).name];
	trimap_url = [trimap_path, img_dir(ii).name];
	img = imread(img_url);
	trimap = imread(trimap_url);
      
    [H, W, C] = size(img);
    
    img_rgb = single(reshape(img, W*H, C));
    trimap_alpha = single(reshape(trimap, W*H, 1));
    
    %% small cut function
    %for jj = 1:cut_size*cut_size     
    for jj = 1
        mask = zeros(cut_size*cut_size, 1, 'uint8'); % [0, 0, 0, 0]
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
        
        % brute force
        F_ind = find(small_trimap == 255); 
        B_ind = find(small_trimap == 0); 
        U_ind = find(small_trimap == 128); 

        img_rgb = single(reshape(small_img, [s_H * s_W, 3]));
        F_rgb = img_rgb(F_ind, :);
        B_rgb = img_rgb(B_ind, :);
        U_rgb = img_rgb(U_ind, :);

        [F_y, F_x] = ind2sub(size(small_trimap), F_ind);
        [B_y, B_x] = ind2sub(size(small_trimap), B_ind);
        [U_y, U_x] = ind2sub(size(small_trimap), U_ind);
        F_s = [F_y, F_x];
        B_s = [B_y, B_x];
        U_s = [U_y, U_x];

        F_mindist = bwdist(small_trimap == 255);
        F_mindist = F_mindist(U_ind);
        B_mindist = bwdist(small_trimap == 0);
        B_mindist = B_mindist(U_ind);    
        
        F = zeros(length(U_ind), 1);
        B = zeros(length(U_ind), 1);
        alpha = zeros(length(U_ind), 1);
        fitness = zeros(length(U_ind), 1);
        
        for uu = 1:length(U_ind)
            F_best = 0;
            B_best = 0;
            alpha_best = 0;
            fitness_best = 1e10;
            for ff = 1:length(F_ind)
                for bb = 1:length(B_ind)
                    Fx_Bx_rgb = F_rgb(ff, :) - B_rgb(bb, :);
                    est_alpha = sum((U_rgb(uu, :) - B_rgb(bb, :)) .* Fx_Bx_rgb, 2) ./ (sum(Fx_Bx_rgb .* Fx_Bx_rgb, 2) + 1);
                    est_alpha(est_alpha > 1) = 1;
                    est_alpha(est_alpha < 0) = 0;
                    % Chromatic distortion
                    cost_c  = norm2(U_rgb(uu, :) - (est_alpha .* F_rgb(ff, :) + (1 - est_alpha) .* B_rgb(bb, :)));
                    % Spatial cost
                    cost_sF = norm2(F_s(ff, :)-U_s(uu, :)) ./ F_mindist(uu, :);
                    cost_sB = norm2(B_s(bb, :)-U_s(uu, :)) ./ B_mindist(uu, :);
                    cost_cd = norm2(Fx_Bx_rgb);
                    current_fitness = (cost_c + cost_sF + cost_sB);                   
                    
                    if current_fitness < fitness_best
                        fitness_best = current_fitness;
                        F_best = ff;
                        B_best = bb;
                        alpha_best = est_alpha;
                        fprintf('U:(%d/%d),F:(%d/%d),B:(%d/%d),%d', uu, length(U_ind), ff, length(F_ind), bb,length(B_ind), current_fitness);
                        fprintf(',change');
                        fprintf('\n')
                    end
                end
            end
            F(uu) = F_best;
            B(uu) = B_best;
            alpha(uu) = alpha_best;
            fitness(uu) = fitness_best;
        end
        alpha_matte = reshape(small_trimap, s_H * s_W, 1);
        alpha_matte(U_ind) = alpha;
        alpha_matte = reshape(alpha_matte, s_H, s_W, 1);
        
        file_name = sprintf('./result/bf-small/%d-%s-%d-%d', max_fitness_evaluation, img_dir(ii).name(1:end-4), cut_size, jj);
        imwrite(alpha_matte, [file_name, '.png']);
        save([file_name, '.mat'], 'F', 'B', 'fitness');
    end
end