% img_path = 'F:\360云盘\研二下\实验\BSR\BSDS500\data\images\test/';
% pb_path = 'F:\360云盘\研二下\实验\BSR\BSDS500\data\images\pbTest/';
img_path = 'F:\360云盘\研三\论文改进\code\newTextureFeature\4/';
img_path2 = 'F:\360云盘\研三\论文改进\code\newTextureFeature\20/';
output_path = 'F:\360云盘\研三\论文改进\code\merge\ms_texture/';
imgs = dir([img_path,'*.png']);
imgs2 = dir([img_path,'*.png']);
% pbs = dir([pb_path,'*.png']);
parfor m = 1:length(imgs)
%     imgs(m).name = '5096.png';
    texture = imread([img_path,imgs(m).name]);
    texture2 = imread([img_path2,imgs2(m).name]);
    [h,w] = size(texture);
    block_size = 12;
    half_block_size = block_size/2;
    texture_feature = zeros(h,w,3);
    
    for i = 1:h
        for j = 1:w
            roi = texture(max(1,i-half_block_size):min(h,i+half_block_size),...
                max(1,j-half_block_size):min(w,j+half_block_size));
            texture_feature(i,j,2) = nnz(roi);
        end
    end
    block_size2 = 8;
    half_block_size2 = block_size2/2;
    for i = 1:h
        for j = 1:w
            roi = texture(max(1,i-half_block_size):min(h,i+half_block_size),...
                max(1,j-half_block_size):min(w,j+half_block_size));
            texture_feature(i,j,3) = std(double(roi(:)));
        end
    end
    texture(texture2 == 0) = 255;
    texture_feature(:,:,1) = texture;
    texture_feature(:,:,1) = normalise(texture_feature(:,:,1));
    texture_feature(:,:,2) = normalise(texture_feature(:,:,2));
    texture_feature(:,:,3) = normalise(texture_feature(:,:,3));
    imwrite(texture_feature,[output_path,imgs(m).name]);
%     I = im2double(gray2rgb(I));
%     pb = imread([pb_path,pbs(m).name]);
%     [fimage, segs, modes, regSize, grad, conf] = edison_wrapper(texture_feature, [],...
%         'SpatialBandWidth',65,'RangeBandWidth',60,'MinimumRegionArea',500);
%     labels_bk = segs;
%     [fimage, segs, modes, regSize, grad, conf] = edison_wrapper(I, @RGB2Luv,...
%         'SpatialBandWidth',8,'RangeBandWidth',6.5,'MinimumRegionArea',100);
%     labels_bk = segs;
%     max_label = max(segs(:))
%     s = regionprops(segs,'Area','Image','BoundingBox');
%     label_area = cat(1,s.Area);
%     label_bbox = round(cat(1,s.BoundingBox));
%     avg_area = mean(label_area);
%     small_area_labels = find(label_area<avg_area*0.5);
%     for i =1:length(small_area_labels)
% %         i
%         idx = small_area_labels(i);
%         l = label_bbox(idx,1);
%         r = label_bbox(idx,1)+label_bbox(idx,3)-1;
%         t = label_bbox(idx,2);
%         b = label_bbox(idx,2)+label_bbox(idx,4)-1;
%         roi = segs(t:b,l:r);
%         roi(s(idx).Image) = max_label;
%         segs(t:b,l:r) = roi;
%     end
%     subplot(2,2,1),imshow(I,[]);
%     subplot(2,2,2),imshow(pb);%imshow(label2rgb(labels,'jet','w','shuffle'),[]);
%     subplot(2,2,3),imshow(labels == max_label);
%     subplot(2,2,4),imshow(label2rgb(labels_bk,'jet','w','shuffle'),[]);
%     imwrite(label2rgb(labels_bk,'jet','w','shuffle'),[output_path,imgs(m).name]);
%     save([output_path,pbs(m).name(1:end-4),'.mat'],'segs');
end