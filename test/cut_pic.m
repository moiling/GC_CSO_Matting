%
% 按照参数剪切图片相应位置，原图添加框框
%
clear;
warning('off');

data_path = '../data/train/';
out_path = '../result/eval-5.000000e+03-mask-11/';
img_path = [data_path, 'input/input_training_highres/'];
trimap_path = [data_path, 'trimap/trimap_training_highres/Trimap1/'];
gt_path = [data_path, 'ground_truth/gt_training_highres/'];
img_dir = dir([img_path, '*.png']);
% y1,y2,x1,x2
data = [
    1271,1570,531,1030;
    21,500,1441,2240;
    101,400,1201,1700;
    751,1050,2021,2520;
    601,1200,451,1450;
    1141,1290,951,1200;
    1841,2140,1751,2250;
    1351,1500,2051,2300
];

for ii = 1
ind = 1;
for m = [5, 11, 14, 15, 16, 17, 25, 27]
    y1 = data(ind, 1);
    y2 = data(ind, 2);
    x1 = data(ind, 3);
    x2 = data(ind, 4);
    disp(img_dir(m).name);
    img_url = [img_path, img_dir(m).name];
    trimap_url = [trimap_path, img_dir(m).name];

    file_name = sprintf('GT%02d_iter_1_without_smoothing.png', m);
    gc_cso_alpha_url = [out_path, 'cc2-cso/', file_name];
    cc_pso_alpha_url = [out_path, 'cc-pso/', file_name];
    cso_alpha_url = [out_path, 'no-cc-cso/', file_name];
    cc_de_s_alpha_url = [out_path, 'cc-de-s/', file_name];

    img = imread(img_url);
    trimap = imread(trimap_url);
    gc_cso_alpha_matte = imread(gc_cso_alpha_url);
    cc_pso_alpha_matte = imread(cc_pso_alpha_url);
    cso_alpha_matte = imread(cso_alpha_url);
    cc_de_s_alpha_matte = imread(cc_de_s_alpha_url);

    origin_mark = img;
    for jj = 1:6
        origin_mark(y1:y2, x1, :) = repmat([255,255,0], y2-y1+1, 1);
        origin_mark(y1:y2, x1+jj, :) = repmat([255,255,0], y2-y1+1, 1);
        origin_mark(y1:y2, x2, :) = repmat([255,255,0], y2-y1+1, 1);
        origin_mark(y1:y2, x2-jj, :) = repmat([255,255,0], y2-y1+1, 1);
        origin_mark(y1, x1:x2, :) = reshape(repmat([255,255,0], x2-x1+1, 1), [1, x2-x1+1, 3]);
        origin_mark(y1+jj, x1:x2, :) = reshape(repmat([255,255,0], x2-x1+1, 1), [1, x2-x1+1, 3]);
        origin_mark(y2, x1:x2, :) = reshape(repmat([255,255,0], x2-x1+1, 1), [1, x2-x1+1, 3]);
        origin_mark(y2-jj, x1:x2, :) = reshape(repmat([255,255,0], x2-x1+1, 1), [1, x2-x1+1, 3]);
    end
    trimap_cut = trimap(y1:y2,x1:x2,:);
    gc_cso_cut = gc_cso_alpha_matte(y1:y2,x1:x2,:);
    cc_pso_cut = cc_pso_alpha_matte(y1:y2,x1:x2,:);
    cso_cut = cso_alpha_matte(y1:y2,x1:x2,:);
    cc_de_s_cut = cc_de_s_alpha_matte(y1:y2,x1:x2,:);
    
    name = sprintf('./result/cut/GT%02d', m);
    imwrite(origin_mark, [name, '_img.png']);
    imwrite(trimap_cut, [name, '_trimap.png']);
    imwrite(gc_cso_cut, [name, '_gccso.png']);
    imwrite(cc_pso_cut, [name, '_ccpso.png']);
    imwrite(cso_cut, [name, '_cso.png']);
    imwrite(cc_de_s_cut, [name, '_ccdes.png']);
    ind = ind + 1;
end
end