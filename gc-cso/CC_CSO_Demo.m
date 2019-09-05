clear;
addpath(genpath(pwd));

path = './data/train/';
img_path = [path,'input/input_training_highres/'];
trimap_path = [path,'trimap/trimap_training_highres/Trimap2/'];
gt_path = [path,'ground_truth/gt_training_highres/'];
img_dir = dir([img_path,'*.png']);
output_path = './result/high-cc-pre-5e3/';
max_fitness_evaluation = 5e3;

if ~exist(output_path,'dir')
    mkdir(output_path);
end
    
%for m = 1:length(img_dir)
for mask_iter = 15:18
for m = [1, 4]
    %profile on -memory
    tic
    disp(img_dir(m).name);
    % ¶ÁÈ¡Í¼Æ¬ºÍÈý·ÖÍ¼
    img = imread([img_path,img_dir(m).name]);
    trimap = imread([trimap_path,img_dir(m).name]);
    MaskAct = TrimapExpansion(double(img), trimap, mask_iter);
    MaskAct(MaskAct == 1) = 255;
    MaskAct(MaskAct == 5) = 0;
    MaskAct(MaskAct == 3) = 128;
    MaskAct = uint8(MaskAct);
    imwrite(MaskAct, [output_path, img_dir(m).name(1:end-4), '_mask_', num2str(mask_iter), '.png']);
    
    % ·Ö±ð»ñÈ¡Ç°¾°¡¢±³¾°¡¢Î´ÖªÇøÓòÏñËØµãµÄÏÂ±ê£¨ÏòÁ¿£©
    F_ind = find(MaskAct == 255);    % Ç°¾°
    B_ind = find(MaskAct == 0);      % ±³¾°
    U_ind = find(MaskAct == 128);    % Î´ÖªÇøÓò
    % »ñÈ¡Í¼ÏñÑÕÉ«£¨[W, H, 3] -> [W*H, 3]£©
    img_rgb = single(reshape(img,[numel(trimap),3]));
    % ·Ö±ð»ñÈ¡Ç°¾°¡¢±³¾°¡¢Î´ÖªÇøÓò¶ÔÓ¦µÄÑÕÉ«
    F_rgb = img_rgb(F_ind,:);
    B_rgb = img_rgb(B_ind,:);
    U_rgb = img_rgb(U_ind,:);
    clear('img_rgb');
    
    % ÑÕÉ«¾ÛÀà
    [cc_rgb, U_cc_ind, cc_U_ind] = color_clustering(U_rgb);
    fprintf('¾ÛÀàÇ°Î¬¶È:%d£¬¾ÛÀàºóÎ¬¶È:%d\n', size(U_rgb, 1), size(cc_rgb, 1));
    % ·Ö±ð»ñÈ¡Ç°¾°¡¢±³¾°¡¢¾ÛÀàºóÎ´ÖªÇøÓò¶ÔÓ¦µÄ¿Õ¼ä×ø±ê£¨[x, y]£©
    F_s = ind2sub(size(trimap), F_ind);
    B_s = ind2sub(size(trimap), B_ind);
    CC_U_s = ind2sub(size(trimap), U_ind(cc_U_ind));
    
    % ·Ö±ð¼ÆËã¾ÛÀàºóUµ½FºÍBµÄ¿Õ¼ä×îÐ¡¾àÀë  
    F_mindist = bwdist(MaskAct == 1);F_mindist = F_mindist(U_ind(cc_U_ind));
    B_mindist = bwdist(MaskAct == 5);B_mindist = B_mindist(U_ind(cc_U_ind));

    %% options for ga
    FitnessFcn = @(x) CSO_CostFunc(x, F_rgb, B_rgb, cc_rgb, F_s, B_s, CC_U_s, F_mindist, B_mindist);
    numberOfVariables = 2 * length(U_ind(cc_U_ind)); % Number of decision variables
    lb = ones(1, numberOfVariables); % Lower bound
    ub = [repmat(size(F_rgb, 1), 1, length(U_ind(cc_U_ind))) repmat(size(B_rgb,1), 1, length(U_ind(cc_U_ind)))]; % Upper bound

    %% run GA
    [x, fval, F] = MyCSO(FitnessFcn, numberOfVariables, lb, ub, max_fitness_evaluation);
    
    %% disp MOGA
    [est_fitness, est_alpha] = CSO_CostFunc_all_fitness(x, F_rgb, B_rgb, cc_rgb, F_s, B_s, CC_U_s, F_mindist, B_mindist);
    clear('F_rgb', 'B_rgb', 'U_rgb', 'F_s', 'B_s', 'U_s', 'F_mindist', 'B_mindist');
    est_alpha(est_alpha>1) = 1;
    est_alpha(est_alpha<0) = 0;
    
    % ÒòÎªËã³öÀ´µÄalphaÊÇÒ»ÑùµÄ£¬Ö±½Ó½«alphaÀ©³ä
    alpha = ones(length(U_ind), 1);

    for ii = 1 : length(U_ind(cc_U_ind))
        alpha(U_cc_ind == ii) = est_alpha(ii);
    end
    
    toc
    
    % ÕâÀïËã³öÀ´µÄxÊÇ¾ÛÀàºóµÄ½á¹û£¬Òª½«ÆäÀ©Õ¹µ½Õû¸öUÇøÓò
    x_ = ones(1, 2 * length(U_ind));
    for ii = 1 : length(U_ind(cc_U_ind))
        ind = find(U_cc_ind == ii);
        x_(ind) = x(ii);
        x_(ind + length(U_ind)) = x(ii + length(U_ind(cc_U_ind)));
    end
    
    % À©³äfitness£¬¸øÖ®ºóÆ½»¬ÓÃ£¬ÕâÀï²»ÄÜÖØÐÂ¼ÆËãfitness£¬²»È»¿Õ¼ä²î¾à»áÌØ±ð´ó
    % ÕâÀï¿ÉÒÔ°ÑFºÍBÒ²¾ÛÀàÁË£¬ÔÚÍ¬ÀàµÄFºÍBÖÐÕÒ×î½üµÄÇófitness
    fitness = ones(length(U_ind), 1);
    for ii = 1 : length(U_ind(cc_U_ind))
        fitness(U_cc_ind == ii) = est_fitness(ii);
    end   
    
    % save alpha matte
    MaskAct(U_ind) = alpha * 255;
    imwrite(MaskAct, [output_path, img_dir(m).name(1:end-4), '_alphamatte_', num2str(mask_iter), '.png']);
    
    % ·ÅÔÚºóÃæ¼ÓÔØ£¬ºÍ±ê×¼Í¼Æ¬½øÐÐ±È½Ï£¬¼ÆËãÎó²îÖµ
    gt = im2single(imread([gt_path,img_dir(m).name]));
    U_gt = gt(trimap == 128);  % ÕâÀïÒªÖØÐÂËãÒ»ÏÂ,ÒòÎªÖ®Ç°µÄU_ind±ÈÎ»ÖÃÇøÓòÒªÐ¡
    pso_alpha_error = sum(abs(double(MaskAct(trimap == 128)) - U_gt));

    %% disp ga
    CSO_data = [round(x), fval, pso_alpha_error];

    filename = sprintf('%s_mask_%d_.mat', img_dir(m).name(1:end-4), mask_iter);
    save([output_path, filename], 'CSO_data', 'x_', 'F', 'fitness');
    
    %% profile
    %p = profile('info');
    %profsave(p, [output_path, 'profile/', filename]);
end
end

%profile off