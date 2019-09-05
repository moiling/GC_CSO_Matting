%
% 聚类的平滑处理操作，confidence使用的是计算出来的fitness
% 单独写了一个文件是因为之前一步聚类后，fitness只对应每一类，而没对应到每个点，这里做了扩充
%
clear;
warning('off');

data_path = './data/train/';
out_path = './result/test/';
mask_path = [out_path, 'mask/'];
img_path = [data_path, 'input/input_training_highres/'];
trimap_path = [data_path, 'trimap/trimap_training_highres/Trimap1/'];
gt_path = [data_path, 'ground_truth/gt_training_highres/'];
img_dir = dir([img_path, '*.png']);

%for m = 1:length(img_dir)
for m = 11
    tic;
    disp(img_dir(m).name);
    img_url = [img_path, img_dir(m).name];
    %trimap_url = [trimap_path, img_dir(m).name];
    trimap_url = [mask_path, img_dir(m).name];
    file_name = sprintf('GT%02d_iter_1_without_smoothing.png', m);
    sub_dir = 'GC-CSO-11/';
    alpha_url = [out_path, sub_dir, file_name];
    
    img = imread(img_url);
    trimap = imread(trimap_url);
    alpha_matte = imread(alpha_url);

    U_ind = find(trimap == 128);
    
    pic_name = img_dir(m).name(1:end-4);
    
    mat_file_name = sprintf('GT%02d_iter_1.mat', m);
    load([out_path, sub_dir, mat_file_name]);
    %% Post-processing which is mentioned in paper titled "A Global Sampling Method for Alpha Matting"
    trimap = trimap(:, :, 1);
    alpha_matte = alpha_matte(:, :, 1);
    [W, H, ~] = size(img);

    trimap = reshape(trimap, W, H, 1);
    img = reshape(img, W, H, 3);
    img = double(img);
    confidence = ones(W * H, 1);

    % fitness(fitness == max(fitness)) = 0;
    fitness = fitness(:, 1:length(U_ind)); 
    fitness = (fitness - min(fitness)) ./ (max(fitness) - min(fitness));

    fitness = reshape(fitness, size(fitness, 2), 1);
    confidence(U_ind) = fitness;
    confidence = reshape(confidence, W, H, 1);
    confidence = confidence .* 255;

    alpha_matte = smoothing(img, alpha_matte, confidence, trimap);
    
    toc;
    
    %out_url = [smoothing_out_path, pic_name];
    %imwrite(alpha_matte, [out_url, '.png']);
    imwrite(alpha_matte, [out_path, sub_dir, 'smoothing/old_', file_name]);
    
    gt = im2single(imread([gt_path,img_dir(m).name]));
    U_gt = gt(U_ind);
    pso_alpha_error = sum(abs(alpha_matte(U_ind)-U_gt));
    fprintf('%d\n', pso_alpha_error);
end