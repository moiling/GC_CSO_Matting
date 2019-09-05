clear
tic
path = './data/train/';
img_path = [path,'input/input_training_lowres/'];
trimap_path = [path,'trimap/trimap_training_lowres/Trimap1/'];
gt_path = [path,'ground_truth/gt_training_lowres/'];
img_dir = dir([img_path,'*.png']);
max_fitness_evaluation = 5e3;
output_path = ['./result/eval-', num2str(max_fitness_evaluation, '%e'), '-mask-', num2str(11), '/'];
mask_path = [output_path, 'mask/'];

data = zeros(27, 3);
for m = 1:length(img_dir)
    disp(img_dir(m).name);

    img = imread([img_path,img_dir(m).name]);
    trimap = imread([trimap_path,img_dir(m).name]);
    mask = TrimapExpansion(double(img), trimap, 11);

    F_ind = find(mask == 255);  
    B_ind = find(mask == 0);    
    U_ind = find(mask == 128); 
    data(m, :) = [length(U_ind), length(F_ind), length(B_ind)];
end