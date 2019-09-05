%
% 平滑处理操作，confidence使用的是计算出来的fitness
%
clear;
warning('off');

data_path = './data/train/';
out_path = './high/5e3/';
img_path = [data_path, 'input/input_training_highres/'];
trimap_path = [data_path, 'trimap/trimap_training_highres/Trimap2/'];
% gt_path = [path, 'gt_training_lowres/'];
img_dir = dir([img_path, '*.png']);
smoothing_out_path = './high/5e3/smoothing/';

%for m = 1:length(img_dir)
for m = [1, 3, 9, 12, 16]
    disp(img_dir(m).name);
    img_url = [img_path, img_dir(m).name];
    trimap_url = [trimap_path, img_dir(m).name];
    
    img = imread(img_url);
    trimap = imread(trimap_url);
    
    F_ind = find(trimap == 255);
    B_ind = find(trimap == 0);
    U_ind = find(trimap == 128);
    
    img_rgb = single(reshape(img,[numel(trimap),3]));

    F_rgb = img_rgb(F_ind,:);
    B_rgb = img_rgb(B_ind,:);
    U_rgb = img_rgb(U_ind,:);
    clear('img_rgb');
    
    F_s = ind2sub(size(trimap), F_ind);
    B_s = ind2sub(size(trimap), B_ind);
    U_s = ind2sub(size(trimap), U_ind);

    F_mindist = bwdist(trimap == 255);F_mindist = F_mindist(U_ind);
    B_mindist = bwdist(trimap == 0);B_mindist = B_mindist(U_ind);
    
    pic_name = img_dir(m).name(1:end-4);
    
    load([out_path, pic_name, '.mat']);

    x = CSO_data(1 : end-2);
    
    [fitness, est_alpha] = ...
    CSO_CostFunc_all_fitness(x, F_rgb, B_rgb, U_rgb, F_s, B_s, U_s, F_mindist, B_mindist);
    clear('F_rgb', 'B_rgb', 'U_rgb', 'F_s', 'B_s', 'U_s', 'F_mindist', 'B_mindist');
    
    est_alpha(est_alpha>1) = 1;
    est_alpha(est_alpha<0) = 0;
    
    alpha_matte = trimap;
    alpha_matte(U_ind) = est_alpha * 255;
    %% Post-processing which is mentioned in paper titled "A Global Sampling Method for Alpha Matting"
    trimap = trimap(:, :, 1);
    [W, H, ~] = size(img);

    trimap = reshape(trimap, W, H, 1);
    img = reshape(img, W, H, 3);
    img = double(img);
    confidence = ones(W * H, 1);

    % fitness中会有最大的10000，要把它排除掉，不然归一化就没效果了
    % fitness(fitness == max(fitness)) = 0;
    fitness = (fitness - min(fitness)) ./ (max(fitness) - min(fitness));

    confidence(U_ind) = fitness;
    confidence = reshape(confidence, W, H, 1);
    confidence = confidence .* 255;

    alpha_matte = smoothing(img, alpha_matte, confidence, trimap);
    
    out_url = [smoothing_out_path, pic_name];
    imwrite(alpha_matte, [out_url, '.png']);
end