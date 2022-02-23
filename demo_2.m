%% set parameters
clear;
close all
clc


% enter file path and file name
folder_path = uigetdir('*.mat', 'Select folder'); folder_path = [folder_path,'/'];
file_name = 'PA-Tet-ON=Luminescence.tif';

p1 = .95; % threshold for first binalization
p2 = .25; % threshold for removing objects much smaller than normal cell size
kernelsize = 5; % kernel size for filter image


flexROIflag = 1; % set 0, if use fixed size ROI
if ~flexROIflag
    margin = 5; % set fixed ROI size
end

champFlag = 1; % set 1, if use selected ROIs
NumChamps = 10; % # of selected ROIs


%% load data
% get movie info
info = imfinfo([folder_path,file_name]);
NumFrames = length(info);
width = info(1).Width; height = info(1).Height;
F = cellfun(@(x) imread([folder_path,file_name],x),num2cell(1:NumFrames), 'uniform',0); % read raw frames



%% make projection image
a = reshape(F,1,1,NumFrames); a = cat(3,a{:});
PRJ = max(a,[],3); % projection image
% PRJ = sum(a,3); % projection image
% PRJ = median(a,3); % projection image

PRJ = PRJ./max(PRJ(:)); % normalize
PRJ = imadjust(PRJ); % saturate contrast
% p = histeq(p); % equalize histogram
PRJ = ordfilt2(PRJ, 5, ones(3)); % median filter



%% binarize projection image
threshold =  quantile(PRJ(:), p1); % quantile at p
BW = im2bw(PRJ, threshold); % binarized image

% bw = im2bw(p, graythresh(p)); % auto thresholding

% bw = wiener2(bw, [kernelsize,kernelsize]); % wiener filter
BW = ordfilt2(BW, 1, ones(kernelsize)); % minimum filter

% get distribution of cell size
stats = regionprops('table', BW, 'FilledArea');
a = stats.FilledArea; % cell size distribution
% get quantile of the cellsize distribution at p
threshold =  ceil(quantile(a, p2));
% remove all object containing fewer than threshold pixels
BW = bwareaopen(BW,threshold);

% fill a gap
% bw = imclose(bw,strel('disk',3));

% fill any holes, so that regionprops can be used to estimate
% the area enclosed by each of the boundaries
% bw = imfill(bw,'holes');



%% object detection and feature extraction
stats = regionprops('table', BW, 'Area','BoundingBox','Centroid');

if champFlag
    [~,I] = sort(stats.Area, 'descend'); % sort by area size
    stats = stats(I(1:NumChamps),:);
end

NumCells = size(stats,1);
ROI = stats.BoundingBox+ones(NumCells,1)*[.5,.5,0,0];



%% make filter image
ROI = mat2cell(ROI, ones(NumCells,1),4);
[X,Y] = meshgrid(1:width,1:height);
TF = cellfun(@(x) ((X>=x(1))&(X<=(x(1)+x(3))))&...
    ((Y>=x(2))&(Y<=(x(2)+x(4)))), ROI, 'uniform',0); % filter image

if flexROIflag % make filter image corresponding to boundary box
    g = stats.Centroid;
    TF = repmat(TF,1,NumFrames);
else % make filter image corresponding to manually defined image    
    % find linear indices of maximum intensity within each boundary box
    [h,w] = cellfun(@(x) find((PRJ.*x)==max(max(PRJ.*x))), TF, 'uniform',0);
    % get representative value of linear indices
    g = cellfun(@(x,y) median([x,y],1), w,h, 'uniform',0);
    TF = repmat(cellfun(@(x) ((X>=x(1)-(margin-1)/2)&(X<=(x(1)+(margin-1)/2)))&...
        ((Y>=x(2)-(margin-1)/2)&(Y<=(x(2)+(margin-1)/2))), g, 'uniform',0), 1,NumFrames);
end



%% get pixel intensity
% each pixel intensity within each bounding box
V = cellfun(@(x,y) y(x), TF, repmat(F,NumCells,1), 'uniform',0); % row is cell#, col is frame#
% get maximum pixel intensity within each bounding box
V_max = cellfun(@max,V);
% average and quantile of pixcel intensities across cells
V_mu = median(V_max);
V_mu = smooth(V_mu); % moving average
V_q = quantile(V_max,[.25,.75],1);
V_q = [smooth(V_q(1,:))';smooth(V_q(2,:))'];


%% visualize
h = figure('Position',[100,10,900,600]);
colormap(h,'gray')

subplot(2,3,1)
imagesc(PRJ);
title('Raw projection image')
set(gca, 'XTickLabel','', 'YTickLabel','')
axis square

subplot(2,3,2)
imagesc(BW)
title('Binalized')
set(gca, 'XTickLabel','', 'YTickLabel','')
axis square

subplot(2,3,3)
imagesc(PRJ)
hold on
if flexROIflag
    cellfun(@(x) rectangle('Position',x,'EdgeColor','r'), ROI, 'uniform',0);
else
    cellfun(@(x) rectangle('Position', [x(1)-1,x(2)-1,margin-1,margin-1],...
        'EdgeColor','r'), g, 'uniform',0);
end
plot(-100,-100,'rs');
axis square
set(gca, 'XLim',[1,width], 'YLim',[1,height], 'XTickLabel','', 'YTickLabel','')
legend('Boundary box');
title('ROI')
hold off

subplot(2,3,4:6)
t = 1:NumFrames;
patch([t,fliplr(t)],[V_q(2,:),fliplr(V_q(1,:))],[1,0,0],...
    'FaceAlpha',.3,'EdgeColor','none')
hold on
plot(t,V_mu, 'r-', 'LineWidth',2);
ylim = get(gca,'YLim');
set(gca, 'YLim',[0,ylim(2)], 'XLim',[0,NumFrames])
xlabel('Frames')
ylabel('Pixel intensity')
hold off