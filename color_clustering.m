function [cc_rgb, U_cc_ind, cc_U_ind] = color_clustering(U_rgb)

% 颜色的所有种类
cc_rgb = unique(U_rgb, 'rows');
% 未知区域中像素点对应颜色种类的下标
[~, U_cc_ind] = ismember(U_rgb, cc_rgb, 'rows');

% 目前没有定义选取一类中哪个像素点，故使用如下方法选取一个点
[~, cc_U_ind] = ismember(cc_rgb, U_rgb, 'rows');

end

