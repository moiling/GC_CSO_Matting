%
% 其实只是一个imshow去白边的方法
%
img_path = '../result/cut/';
img_dir = dir([img_path, '*.png']);

for ii = 1:length(img_dir)
    
    figure(ii);
    i = imread([img_path, img_dir(ii).name]);
    imshow(i, 'border', 'tight');

end