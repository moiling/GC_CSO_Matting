%
% 按照参数剪切图片相应位置，原图添加框框
% 保存为eps格式
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

data2 = [1451, 1600, 2101, 2350];

for ii = 1
ind = 1;
%for m = [5, 11, 14, 15, 16, 17, 25, 27]
for m = 25
    y1 = data2(ind, 1);
    y2 = data2(ind, 2);
    x1 = data2(ind, 3);
    x2 = data2(ind, 4);
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
    for jj = 1:10
        origin_mark(y1:y2, x1, :) = repmat([0,0,255], y2-y1+1, 1);
        origin_mark(y1:y2, x1+jj, :) = repmat([0,0,255], y2-y1+1, 1);
        origin_mark(y1:y2, x2, :) = repmat([0,0,255], y2-y1+1, 1);
        origin_mark(y1:y2, x2-jj, :) = repmat([0,0,255], y2-y1+1, 1);
        origin_mark(y1, x1:x2, :) = reshape(repmat([0,0,255], x2-x1+1, 1), [1, x2-x1+1, 3]);
        origin_mark(y1+jj, x1:x2, :) = reshape(repmat([0,0,255], x2-x1+1, 1), [1, x2-x1+1, 3]);
        origin_mark(y2, x1:x2, :) = reshape(repmat([0,0,255], x2-x1+1, 1), [1, x2-x1+1, 3]);
        origin_mark(y2-jj, x1:x2, :) = reshape(repmat([0,0,255], x2-x1+1, 1), [1, x2-x1+1, 3]);
    end
    temp = [0, 0, 80];
    temp = uint8(temp);
    mark = reshape(repmat(temp, (y2-y1+1) * (x2-x1+1), 1), [y2-y1+1, x2-x1+1, 3]);
    origin_mark(y1:y2,x1:x2,:) = origin_mark(y1:y2,x1:x2,:) + mark;
    
    trimap_cut = trimap(y1:y2,x1:x2,:);
    gc_cso_cut = gc_cso_alpha_matte(y1:y2,x1:x2,:);
    cc_pso_cut = cc_pso_alpha_matte(y1:y2,x1:x2,:);
    cso_cut = cso_alpha_matte(y1:y2,x1:x2,:);
    cc_de_s_cut = cc_de_s_alpha_matte(y1:y2,x1:x2,:);
    image_cut = img(y1:y2,x1:x2,:);
    
    name = sprintf('./result/cut2/GT%02d', m);
    imwrite(origin_mark, [name, '_2_img.png']);
    imwrite(trimap_cut, [name, '_2_trimap.png']);
    imwrite(gc_cso_cut, [name, '_2_gccso.png']);
    imwrite(cc_pso_cut, [name, '_2_ccpso.png']);
    imwrite(cso_cut, [name, '_2_cso.png']);
    imwrite(cc_de_s_cut, [name, '_2_ccdes.png']);
    imwrite(image_cut, [name, '_2_imgcut.png']);
    ind = ind + 1;
    
    % resize
    resize = [150, 250];
    origin_mark = imresize(origin_mark, resize);
    trimap_cut = imresize(trimap_cut, resize);
    gc_cso_cut = imresize(gc_cso_cut, resize);
    cc_pso_cut = imresize(cc_pso_cut, resize);
    cso_cut = imresize(cso_cut, resize);
    cc_de_s_cut = imresize(cc_de_s_cut, resize);
    image_cut = imresize(image_cut, resize);
    
    % save eps
    name = sprintf('./result/cut2/eps/GT%02d', m);
    imshow(origin_mark, 'border', 'tight');
    %saveas(gcf, [name, '_img.eps']);
    print('-depsc', [name, '_2_img.eps']);
    
    imshow(trimap_cut, 'border', 'tight');
    print('-depsc', [name, '_2_trimap.eps']);
    
    imshow(gc_cso_cut, 'border', 'tight');
    print('-depsc', [name, '_2_gccso.eps']);
    
    imshow(cc_pso_cut, 'border', 'tight');
    print('-depsc', [name, '_2_ccpso.eps']);
    
    imshow(cso_cut, 'border', 'tight');
    print('-depsc', [name, '_2_cso.eps']);
    
    imshow(cc_de_s_cut, 'border', 'tight');
    print('-depsc', [name, '_2_ccdes.eps']);
    
    imshow(image_cut, 'border', 'tight');
    print('-depsc', [name, '_2_imgcut.eps']);
end
end