% 
% 实验GC-CSO、CSO、CC-PSO、CC-DE-S
% 统一读取数据、预处理、保存结果（未做平滑处理）、运行状态（时间、内存等）
%
clear;
addpath(genpath(pwd));

% 所有资源位置，参数
max_fitness_evaluation = 5e3;
mask_iter = 11;
path = './data/train/';
img_path = [path, 'input/input_training_lowres/'];
trimap_path = [path, 'trimap/trimap_training_lowres/Trimap1/'];
gt_path = [path, 'ground_truth/gt_training_lowres/'];
img_dir = dir([img_path, '*.png']);
output_path = ['./result/eval-', num2str(max_fitness_evaluation, '%e'), '-mask-', num2str(mask_iter), '/'];
gc_cso_path = [output_path, 'cc2-cso/'];
pso_path = [output_path, 'cc-pso/'];
de_path = [output_path, 'cc-de-s/'];
cso_path = [output_path, 'no-cc-cso/'];
mask_path = [output_path, 'mask/'];
run_gc_cso = true;
run_cc_pso = true;
run_cc_de_s = true;
run_cso = true;

% 创建输出目录
if ~exist(output_path, 'dir')
    mkdir(output_path);
    mkdir(gc_cso_path);
    mkdir(pso_path);
    mkdir(de_path);
    mkdir(cso_path);
    mkdir(mask_path);
end

% 挑选出三类图片，依次执行
% 按照维度低、中、高三类顺序，每次挑出类型中降维后维度最小的图算，这样算的快
sorted_pics = [17, 23, 25, 15, 9, 13, 5, 2, 14, 10, 16, 12, 7, 18, 6, 20, 22, 11, 1, 3, 24, 21, 8, 26, 27, 4, 19];
for ii = sorted_pics
	fprintf('%d/27,%s\n', ii, img_dir(ii).name);
    % 读资源
    img_url = [img_path, img_dir(ii).name];
	trimap_url = [trimap_path, img_dir(ii).name];
	img = imread(img_url);
	trimap = imread(trimap_url);
	fprintf('start expansion...\n');
	% 统一做预处理
    if exist([mask_path, img_dir(ii).name(1:end-4), '.png'], 'file')
        mask = imread([mask_path, img_dir(ii).name(1:end-4), '.png']);
        fprintf('already expansioned. read mask finished.\n');
    else
        mask = TrimapExpansion(double(img), trimap, mask_iter);
        imwrite(mask, [mask_path, img_dir(ii).name(1:end-4), '.png']);
        fprintf('expansion finished.\n');
    end

	% 运算30次
    for jj = 1:30
        fprintf('%d/30\n', jj);

		file_name = sprintf('%s_iter_%d', img_dir(ii).name(1:end-4), jj);
		% GC-CSO
        if run_gc_cso && ~exist([gc_cso_path, file_name, '_without_smoothing.png'], 'file')
        profile on -memory
        tic
		[alpha_matte, fitness, x] = GC_CSO(img, trimap, max_fitness_evaluation, mask);
        toc
        % profile
        p = profile('info');
        profsave(p, [gc_cso_path, 'profile/', file_name]);
		% 保存GC-CSO中间结果（待平滑处理） 	
        save([gc_cso_path, file_name, '.mat'], 'alpha_matte', 'fitness', 'x');				
    	imwrite(alpha_matte, [gc_cso_path, file_name, '_without_smoothing.png']);
        end

    	% CC-PSO
        if run_cc_pso && ~exist([pso_path, file_name, '_without_smoothing.png'], 'file')
        profile on -memory
        tic
    	[alpha_matte, fitness, x] = CC_PSO(img, trimap, max_fitness_evaluation, mask);
        toc
        % profile
        p = profile('info');
        profsave(p, [pso_path, 'profile/', file_name]);
    	% 保存CC-PSO中间结果（待平滑处理)	
        save([pso_path, file_name, '.mat'], 'alpha_matte', 'fitness', 'x');				
    	imwrite(alpha_matte, [pso_path, file_name, '_without_smoothing.png']);
        end

        % CC-DE-S 这个直接使用师兄的代码，已经做好了预处理和平滑处理，把师兄的MSE代码删掉，最后统一计算
        if run_cc_de_s && ~exist([de_path, file_name, '_without_smoothing.png'], 'file')
        profile on -memory
        tic
    	[alpha_matte, fitness, Best, FSample, BSample, Un] = CC_DE_S(img, trimap, max_fitness_evaluation, mask);
        toc
        % profile
        p = profile('info');
        profsave(p, [de_path, 'profile/', file_name]);
        save([de_path, file_name, '.mat'], 'alpha_matte', 'fitness', 'Best', 'FSample', 'BSample', 'Un');				
    	imwrite(alpha_matte, [de_path, file_name, '_without_smoothing.png']);
        end
        
        % CSO
        if run_cso && ~exist([cso_path, file_name, '_without_smoothing.png'], 'file')
        profile on -memory
        tic
        [alpha_matte, fitness, x] = CSO(img, trimap, max_fitness_evaluation, mask);
        toc
        % profile
        p = profile('info');
        profsave(p, [cso_path, 'profile/', file_name]);
        % 保存CC-PSO中间结果（待平滑处理)
        save([cso_path, file_name, '.mat'], 'alpha_matte', 'fitness', 'x');             
        imwrite(alpha_matte, [cso_path, file_name, '_without_smoothing.png']);
        end
        % 平滑处理和与标准图的评价交给实验室的电脑做，new_smoothing和test_error
    end
end
profile off
