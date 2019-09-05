%สนำรืขสอ
%imshow(segs,[])
%imshow(segs==1,[])
%[x,y] = find(segs==1)
I = imread('102062.png');
[fimage, segs, modes, regSize, grad, conf] = edison_wrapper(I, []);
labels_bk = segs;
