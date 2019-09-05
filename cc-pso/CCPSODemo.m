clear
path = 'D:\SCUT\YihuiLiang\dataset\train\low_res\';
img_path = [path,'img\'];
trimap_name = 'Trimap1\';
trimap_path = [path,trimap_name];
gt_path = [path,'gt\'];
img_dir = dir([img_path,'*.png']);
parpool('local', 12)
parfor m = 1:length(img_dir)
    disp(img_dir(m).name);
    img = imread([img_path,img_dir(m).name]);
    trimap = imread([trimap_path,img_dir(m).name]);
    gt = im2single(imread([gt_path,img_dir(m).name]));
    if size(gt,3)>1
        gt = rgb2gray(gt);
    end
    F_ind = find(trimap == 255);
    B_ind = find(trimap == 0);
    U_ind = find(trimap == 128);
    U_num = length(U_ind);
    img_rgb = single(reshape(img,[numel(trimap),3]));
    U_gt = gt(U_ind);
    %% 样本属性计算
    % 颜色
    F_rgb = img_rgb(F_ind,:);
    B_rgb = img_rgb(B_ind,:);
    U_rgb = img_rgb(U_ind,:);
    % 距离
    [F_y,F_x] = ind2sub(size(trimap),F_ind); F_yx = single([F_y,F_x]);
    [B_y,B_x] = ind2sub(size(trimap),B_ind); B_yx = single([B_y,B_x]);
    [U_y,U_x] = ind2sub(size(trimap),U_ind); U_yx = single([U_y,U_x]);
    F_s  = [F_y,F_x];
    B_s  = [B_y,B_x];
    U_s  = [U_y,U_x];
    
    % 最小距离
    F_mindist = bwdist(trimap == 255);F_mindist = F_mindist(U_ind);
    B_mindist = bwdist(trimap == 0);B_mindist = B_mindist(U_ind);
    GA_data = zeros(length(U_ind),4);
%     CCPSO_data = zeros(length(U_ind),4);
    

        %% options for ga
        FitnessFcn = @(x) CCPSO_CostFunc( x,F_rgb,B_rgb,U_rgb,F_s,B_s,U_s,F_mindist,B_mindist);
        numberOfVariables = 2*length(U_ind); % Number of decision variables
        lb = [ones(1,numberOfVariables)]; % Lower bound
        ub = [repmat(size(F_rgb,1),1,length(U_ind)) repmat(size(B_rgb,1),1,length(U_ind))]; % Upper bound
        max_fitness_evaluation = 5e3;
        %% run GA
        [ x,fval] = MyCCPSO( FitnessFcn,numberOfVariables,lb,ub,max_fitness_evaluation);
        %% disp MOGA
        [ fitness, est_alpha] = ...
        CCPSO_CostFunc(x,F_rgb,B_rgb,U_rgb,F_s,B_s,U_s,F_mindist,B_mindist);
        est_alpha(est_alpha>1) = 1;
        est_alpha(est_alpha<0) = 0;
        pso_alpha_error = sum(abs(est_alpha-U_gt));

        %% disp ga
        CCPSO_data = [round(x),fval,pso_alpha_error];

    filename = sprintf('%s%s.mat',trimap_name,img_dir(m).name(1:end-4));
    parsave(filename,CCPSO_data);
end