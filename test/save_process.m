%
% 保存过程
%
function save_process(x, FES, small_img, small_trimap, mask, num, FitnessFcn)
    [s_H, s_W, ~] = size(small_img);
    F_ind = find(small_trimap == 255); 
    B_ind = find(small_trimap == 0); 
    U_ind = find(small_trimap == 128); 
    img_rgb = single(reshape(small_img, [s_H * s_W, 3]));
    F_rgb = img_rgb(F_ind, :);
    B_rgb = img_rgb(B_ind, :);
    
    [fitness, est_alpha] = FitnessFcn(x);
    est_alpha(est_alpha > 1) = 1;
    est_alpha(est_alpha < 0) = 0;
    est_alpha = 255 * est_alpha;
    alpha_matte = mask;
    alpha_matte(U_ind) = est_alpha;

    file_name = sprintf('%d-%d', num, FES);
    imwrite(alpha_matte, ['./result/cso-small2/process/', file_name, '.png']);
    save(['./result/cso-small2/process/', file_name, '.mat'], 'x', 'fitness');

    % F,B
    F = reshape(small_img, s_W*s_H, 3);
    B = reshape(small_img, s_W*s_H, 3);
    x = round(x);
    for kk = 1:length(U_ind)
        F(U_ind(kk), :) = F_rgb(x(kk), :);
        B(U_ind(kk), :) = B_rgb(x(kk+length(U_ind)), :);
    end
    F = reshape(F, s_H, s_W, 3);
    B = reshape(B, s_H, s_W, 3);

    imwrite(F, ['./result/cso-small2/process/F/', file_name, '.png']);
    imwrite(B, ['./result/cso-small2/process/B/', file_name, '.png']);
    imwrite(F, ['./result/cso-small2/process/cut/', file_name, '.png'], 'Alpha', double(alpha_matte)/255);

    FF_BB_sample = reshape(small_img, s_H*s_W,3);
    FF_BB_sample(F_ind(x(1:length(U_ind))), :) = repmat([255,255,255], length(U_ind), 1);
    FF_BB_sample(B_ind(x(length(U_ind)+1:end)), :) = repmat([0,0,0], length(U_ind), 1);
    FF_BB_sample = reshape(FF_BB_sample, s_H,s_W,3);
    imwrite(FF_BB_sample, ['./result/cso-small2/process/F-B/', file_name, '.png']);
end

