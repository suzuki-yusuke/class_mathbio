clear;
close all
clc



%% load data


% enter file path and file name
folder_path  = uigetdir('*.mat', 'Select folder'); folder_path = [folder_path,'/'];
file_name = dir([folder_path,'*.tif']); file_name = {file_name.name};

I = cellfun(@(x) imread([folder_path,x]), file_name, 'uniform',0); % read all images
B = I{1}; R = I{2}; G = I{3}; J = cat(3,R(:,:,1),G(:,:,2),B(:,:,3));



%% expression of image data

rgb = J(255:269, 50:64, :); % get small region from RGB image
gry = rgb2gray(rgb); % make grayscale image of the small region
figure;
subplot(1,3,1); image(J); title('original image'); axis square
hold on
rectangle('Position',[50,255,269-255+1,64-50+1],'EdgeColor','r','LineWidth',2)

subplot(1,3,2); image(rgb); title('RGB image'); axis square;
subplot(1,3,3); colormap gray; image(gry); title('grayscale image'); axis square;
disp('RGB expression'); disp(rgb) % show matrix expression of RGB image
disp('grayscale expression'); disp(gry) % show matrix expression of grayscale



%% color histogram on RGB image
edges = 0:5:255;
figure;
subplot(4,4,1); image(R); axis square; title('red image');
xlabel('width (column)'); ylabel('height (row)');
subplot(4,4,2); histogram(R(:,:,1),edges); set(gca,'XLim',[0,255]);
axis square; title('red histogram'); xlabel('intensity'); ylabel('frequency');
subplot(4,4,3); histogram(R(:,:,2),edges); set(gca,'XLim',[0,255]); axis square
axis square; title('green histogram');
subplot(4,4,4); histogram(R(:,:,3),edges); set(gca,'XLim',[0,255]); axis square
axis square; title('blue histogram');

subplot(4,4,5); image(G); axis square; title('green image');
subplot(4,4,6); histogram(G(:,:,1),edges); set(gca,'XLim',[0,255]); axis square
subplot(4,4,7); histogram(G(:,:,2),edges); set(gca,'XLim',[0,255]); axis square
subplot(4,4,8); histogram(G(:,:,3),edges); set(gca,'XLim',[0,255]); axis square

subplot(4,4,9); image(B); axis square; title('blue image');
subplot(4,4,10); histogram(B(:,:,1),edges); set(gca,'XLim',[0,255]); axis square
subplot(4,4,11); histogram(B(:,:,2),edges); set(gca,'XLim',[0,255]); axis square
subplot(4,4,12); histogram(B(:,:,3),edges); set(gca,'XLim',[0,255]); axis square

subplot(4,4,13); image(J); axis square; title('merged image');
subplot(4,4,14); histogram(J(:,:,1),edges); set(gca,'XLim',[0,255]); axis square
subplot(4,4,15); histogram(J(:,:,2),edges); set(gca,'XLim',[0,255]); axis square
subplot(4,4,16); histogram(J(:,:,3),edges); set(gca,'XLim',[0,255]); axis square



%% grayscale image
K = rgb2gray(J);
figure;
colormap gray
subplot(1,2,1); image(K); axis square; title('grayscale image')
subplot(1,2,2); histogram(K,edges); axis square;



%% binarize 1
figure;
colormap gray
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
threshold = 50;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BW = im2bw(K,threshold/255);
subplot(2,2,1); image(K); axis square; title('original image');

subplot(2,2,2); histogram(K,edges); hold on;
ylim = get(gca,'YLim'); h = plot([threshold,threshold],[0,ylim(2)],'r-');
legend(h,{'threshold'}, 'box','off');
axis square; hold off;

subplot(2,2,3); imagesc(BW); axis square; title('binarized image');



%% binalize otsu

figure;
colormap gray
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
threshold = graythresh(K);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BW = im2bw(K,threshold);
subplot(2,2,1); image(K); axis square; title('original image');

subplot(2,2,2); histogram(K,edges); hold on;
ylim = get(gca,'YLim'); h = plot([threshold.*255,threshold.*255],[0,ylim(2)],'r-');
legend(h,{'threshold'}, 'box','off');
axis square; hold off;

subplot(2,2,3); imagesc(BW); axis square; title('binarized image');



%% filter

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kernel = [3,3];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(2,2,4); imagesc(medfilt2(BW,kernel)); axis square; title('binarized and filtered image');



%% adjuct contrast
figure;
colormap gray;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
l = .03; u = .45; % % threshold of lower/upper of intensity histogram
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(2,2,1); image(K); title('original image'); axis square;
subplot(2,2,2); histogram(K,edges); set(gca,'XLim',[0,255]); hold on;
set(gca,'XLim',[0,255],'YLim',[0,8*10^4]); axis square;

subplot(2,2,3); image(imadjust(K,[l,u],[])); title('contrasted image'); axis square;

subplot(2,2,4); histogram(imadjust(K,[l,u],[]),edges);
set(gca,'XLim',[0,255],'YLim',[0,8*10^4]); axis square