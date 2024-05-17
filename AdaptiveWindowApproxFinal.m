function [FilteredImage] = AdaptiveWindowApproxFinal(InputImage,Widx)

% clear;clc;  close all;
lmt = 3;
% [InputImage] = imread('Input/lena_color_512.tif');
InputImage =   im2double(InputImage);
% InputImage = imnoise(InputImage,'gaussian',0,0.07);
disp('Size of the Image');
[m,n,c] = size(InputImage);
RefImage = InputImage; % wind=1;

nbd = ones(3);
GrayImg1 = imopen(InputImage,nbd); GrayImg2 = imclose(InputImage,nbd); InputImage = ((1*InputImage+1*GrayImg1+1*GrayImg2)/3); 
% figure;imshow(InputImage); %
if(c>1)
GrayImg = rgb2gray(InputImage);
else
GrayImg = InputImage;   %(InputImage(:,:,1) + InputImage(:,:,2) + InputImage(:,:,3))/3;
end
GrayImg = im2double(GrayImg);   % InputImage = (imopen(InputImage,nbd)+imclose(InputImage,nbd))/2; % medfilt2(GrayImg,[3,3]); imfilter(InputImage,nbd); %
GrayImg = normalize(GrayImg(1:m*n),'range'); % (test-min(min(test)))/(max(max(test))-min(min(test)));
% GrayImg = ceil(GrayImg*255);
GrayImg = reshape(GrayImg,[m,n]);
GrayImg1 = InputImage; % GrayImg;



Fibo = [2 3 5 8 13 21 34 55]; Len = 55; NbrH_temp = zeros(m*n,111); NbrV_temp = zeros(m*n,111); NbrD1_temp = zeros(m*n,111); NbrD2_temp = zeros(m*n,111);
eps = 0.001;  Morph_GradImg = zeros(m,n); Pxl_SkwRange = zeros(m*n,1); 
for grd=1:1
wind = 0; Pxl_Range = zeros(m*n,1);
for idx=([7 5 3 1]+1)%((4:7)-grd+1)
    w = Fibo(idx);
wind = wind + 1;
W = -w:w; % w1=w; % W1=W;
for bnd=1:c
 GrayImg1 = InputImage(:,:,bnd);   
for pxl = 1:m*n

% [I,J] = ind2sub([m,n],pxl);
if(wind==1)
w1=1; [X,Y] = meshgrid(-w1:w1,-w1:w1);
W1 = -w1:w1;
ImdNbr = pxl + W1*m; ImdNbr = repmat(ImdNbr,2*w1+1,1); ImdNbr = ImdNbr + Y; % ImdNbr = [ImdNbr - w ImdNbr + w];
ImdNbr = ImdNbr(ImdNbr>0 & ImdNbr<=m*n);
Morph_GradImg(pxl) = max(GrayImg1(ImdNbr)) - min(GrayImg1(ImdNbr)) + eps; % max(abs(GrayImg(pxl) - (GrayImg(ImdNbr))));   %   InputImg(pxl)) % 
% Morph_GradImg(I,J) = (max(GrayImg(ImdNbr)) - min(GrayImg(ImdNbr))); %1/(1+exp(-(max(GrayImg1(ImdNbr)) - min(GrayImg1(ImdNbr))))) + eps; % Pxl_Range(pxl);
end

if (wind==1)
   wm=w; Wm = -wm:wm;
   NbrH = pxl + Wm*m;  NbrH(NbrH<=0 | NbrH>m*n)= pxl;
   NbrV = pxl + Wm;  NbrV(NbrV<=0 | NbrV>m*n)= pxl;
   NbrD1 = pxl + Wm*m + Wm;  NbrD1(NbrD1<=0 | NbrD1>m*n)= pxl;
   NbrD2 = pxl + Wm*m - Wm;  NbrD2(NbrD2<=0 | NbrD2>m*n)= pxl;
   
   NbrH_temp(pxl,:) =  NbrH;
   NbrV_temp(pxl,:) =  NbrV;
   NbrD1_temp(pxl,:) =  NbrD1;
   NbrD2_temp(pxl,:) =  NbrD2;
   Len = wm;
else
   wm=w; % Wm = -wm:wm; 
   wmm1 = Len-w+1; wmm2 = Len+w+1;
   NbrH =  NbrH_temp(pxl,wmm1:wmm2);
   NbrV =  NbrV_temp(pxl,wmm1:wmm2);
   NbrD1 =  NbrD1_temp(pxl,wmm1:wmm2);
   NbrD2 =  NbrD2_temp(pxl,wmm1:wmm2);
%    Len = wm;
end

MDH = abs(mean(GrayImg(NbrH(wm:-1:1)))-mean(GrayImg(NbrH(wm+2:end))))+eps;%Entrpy(GrayImg(NbrH));%/sum(Pxl_Range(NbrH).^2);
MDV = abs(mean(GrayImg(NbrV(wm:-1:1)))-mean(GrayImg(NbrV(wm+2:end))))+eps;%Entrpy(GrayImg(NbrV));%/sum(Pxl_Range(NbrV).^2);
MDD1 = abs(mean(GrayImg(NbrD1(wm:-1:1)))-mean(GrayImg(NbrD1(wm+2:end))))+eps;%Entrpy(GrayImg(NbrD1));%/sum(Pxl_Range(NbrD1).^2);
MDD2 = abs(mean(GrayImg(NbrD2(wm:-1:1)))-mean(GrayImg(NbrD2(wm+2:end))))+eps;%Entrpy(GrayImg(NbrD2));%/sum(Pxl_Range(NbrD2).^2);
NbrAll = [NbrH NbrV NbrD1 NbrD2];

if(wind<=4)
Pxl_SkwRange(pxl) = Pxl_SkwRange(pxl)+(((max(Morph_GradImg(NbrAll))-median(Morph_GradImg(NbrAll)))/(max(Morph_GradImg(NbrAll))-min(Morph_GradImg(NbrAll))+eps))); %sum(Morph_GradImg(NbrAll));
end
Pxl_Range(pxl) = (Pxl_Range(pxl) + mean([MDH MDV MDD1 MDD2])); % +  mean([JSDH2 JSDV2 JSDD12 JSDD22])  +.^(1/(wind+1)); % Pxl_Range(pxl)+ /sum([MDH MDV MDD1 MDD2]); % 1/(1+exp(-mean([MDH MDV MDD1 MDD2])));%+Pxl_Range(pxl);
end
end
end
   Pxl_RangeFinal =  Pxl_Range; % 1./(1+exp(-Pxl_Range)); 

end
Pxl_Range = Pxl_SkwRange.*Pxl_RangeFinal+eps; % 1./((1+exp(-Pxl_RangeFinal)).*(1+exp(-Pxl_SkwRange)))+eps;%(Pxl_SkwRange.*Pxl_RangeFinal)+eps;
histfit(Pxl_Range,150,'Lognormal');
InputImage = RefImage;

Wadpt = ones(m,n)*13 ; w=[3 5 8 13 21 34];
wmin = 2;   eps = 0.001; wind = Widx;
EdgeMap = ones(m*n,c);

[phat,~] = mle((Pxl_Range),'Distribution','LogNormal'); 
      Wadpt(Pxl_Range >= exp((phat(1)+(phat(2)^2)/2))) = w(1);
      Wadpt(Pxl_Range >= exp((phat(1))) & Pxl_Range < exp((phat(1)+(phat(2)^1)/1))) = w(wind); 
      Wadpt(Pxl_Range >= exp((phat(1)-phat(2))) & Pxl_Range < exp((phat(1)))) = w(wind+1); 
      Wadpt(Pxl_Range < exp((phat(1)-phat(2)))) = w(6); 

% EdgeMap = reshape(EdgeMap,m*n,[]);
for band =1:c
    GrayImg = InputImage(:,:,band);    
    GrayImg = normalize(GrayImg(1:m*n),'range'); 
    GrayImg=ceil(GrayImg*256);
    
for pxl =  1:m*n    
w = Wadpt(pxl);
W = -w:w;
NbrH = pxl + W*m;  NbrH(NbrH<=0 | NbrH>m*n)= pxl; % GrayImg(pxl)=mean(GrayImg(NbrH)); % Pat_Nbr = [Pat_Nbr NbrH];
LocalPatternH = abs(GrayImg(NbrH) - min([mean(GrayImg(NbrH(1:w))),mean(GrayImg(NbrH(w+2:2*w+1)))]))+eps; % LocalPattern = LocalPattern(:,3);

lp = floor(length(LocalPatternH)/2);
StandaredPattern = zeros(1,lp*2+1);%+ eps/255;
if(GrayImg(pxl) >= (median(GrayImg(NbrH(1:w)))+median(GrayImg(NbrH(w+2:2*w+1))))/2)
    StandaredPattern(w+1)=1;
else
    StandaredPattern(w+1)=0;
end
if(median(LocalPatternH(1:(lp)))<=median(LocalPatternH(((lp)+2):end)))
   StandaredPattern(((lp)+2):end)=1; 
else
   StandaredPattern((1:(lp)))=1; 
end
JSDH1 = JSDiv(LocalPatternH,StandaredPattern);
StandaredPattern =   ones(length(LocalPatternH),1)';
JSDH2 = JSDiv(LocalPatternH,StandaredPattern);


NbrV = pxl + W;  NbrV(NbrV<=0 | NbrV>m*n)= pxl; 
LocalPatternV = abs(GrayImg(NbrV) - min([mean(GrayImg(NbrV(1:w))),mean(GrayImg(NbrV(w+2:2*w+1)))]))+eps; 


lp = floor(length(LocalPatternV)/2);
StandaredPattern = zeros(1,lp*2+1);% + eps/255;
if(GrayImg(pxl) >= (median(GrayImg(NbrV(1:w)))+median(GrayImg(NbrV(w+2:2*w+1))))/2)
    StandaredPattern(w+1)=1;
else
    StandaredPattern(w+1)=0;
end
if(median(LocalPatternV(1:(lp)))<=median(LocalPatternV(((lp)+2):end)))
   StandaredPattern(((lp)+2):end)=1; 
else
   StandaredPattern((1:(lp)))=1; 
end
JSDV1 = JSDiv(LocalPatternV,StandaredPattern);
StandaredPattern =  ones(length(LocalPatternV),1)';
JSDV2 = JSDiv(LocalPatternV,StandaredPattern);

NbrD1 = pxl + W*m + W;  NbrD1(NbrD1<=0 | NbrD1>m*n)= pxl; 
LocalPatternD1 = abs(GrayImg(NbrD1) - min([mean(GrayImg(NbrD1(1:w))),mean(GrayImg(NbrD1(w+2:2*w+1)))]))+eps; % LocalPattern = LocalPattern(:,3);


lp = floor(length(LocalPatternD1)/2);
StandaredPattern = zeros(1,lp*2+1);% + eps/255;
if(GrayImg(pxl) >= (median(GrayImg(NbrD1(1:w)))+median(GrayImg(NbrD1(w+2:2*w+1))))/2)
    StandaredPattern(w+1)=1;
else
    StandaredPattern(w+1)=0;
end
if(median(LocalPatternD1(1:(lp)))<=median(LocalPatternD1(((lp)+2):end)))
   StandaredPattern(((lp)+2):end)=1; %LocalPattern(((lp)+2):end) = mean(LocalPattern(((lp)+2):end));
else
   StandaredPattern((1:(lp)))=1; %LocalPattern((1:(lp))) = mean(LocalPattern(1:(lp)));
end
JSDD11 = JSDiv(LocalPatternD1,StandaredPattern);
StandaredPattern =   ones(length(LocalPatternD1),1)';%/length(LocalPattern);  imcomplement(StandaredPattern); % GrayImg(NbrD1);%
JSDD12 = JSDiv(LocalPatternD1,StandaredPattern);

NbrD2 = pxl + W*m - W;  NbrD2(NbrD2<=0 | NbrD2>m*n)= pxl; % GrayImg(pxl)=mean(GrayImg(NbrD2)); % Pat_Nbr = [Pat_Nbr NbrD2];
LocalPatternD2 = abs(GrayImg(NbrD2) - min([mean(GrayImg(NbrD2(1:w))),mean(GrayImg(NbrD2(w+2:2*w+1)))]))+eps; % LocalPattern = LocalPattern(:,3);


lp = floor(length(LocalPatternD2)/2);
StandaredPattern = zeros(1,lp*2+1);% + eps/255;
if(GrayImg(pxl) >= (median(GrayImg(NbrD2(1:w)))+median(GrayImg(NbrD2(w+2:2*w+1))))/2)
    StandaredPattern(w+1)=1;
else
    StandaredPattern(w+1)=0;
end
if(median(LocalPatternD2(1:(lp)))<=median(LocalPatternD2(((lp)+2):end)))
   StandaredPattern(((lp)+2):end)=1; %LocalPattern(((lp)+2):end) = mean(LocalPattern(((lp)+2):end));
else
   StandaredPattern((1:(lp)))=1; %LocalPattern((1:(lp))) = mean(LocalPattern(1:(lp)));
end
JSDD21 = JSDiv(LocalPatternD2,StandaredPattern);
% LocalPatternD2 =   GrayImg(NbrD2); % abs(GrayImg(NbrD2) - min([median(GrayImg(NbrD2(1:w))),median(GrayImg(NbrD2(w+2:2*w+1)))])); %
StandaredPattern =  ones(length(LocalPatternD2),1)';%/length(LocalPattern);  imcomplement(StandaredPattern); % GrayImg(NbrD2);% 
JSDD22 = JSDiv(LocalPatternD2,StandaredPattern);
 
 
temp1 = [JSDH1 JSDV1 JSDD11 JSDD21]; % temp1 = sort(temp1); 
temp2 = [JSDH2 JSDV2 JSDD12 JSDD22]; 


if(w>=wmin+0 && min(temp1(1:4)) >= mean(temp2)) 
    EdgeMap(pxl,band) = 0;
end
end
end
size(EdgeMap)
EdgeMap = reshape(EdgeMap,[m,n,c]); 
BW = max(EdgeMap,[],3)>=1; 
% figure;
% imshow(mat2gray((BW)));
GrayImg = GrayImg1;
lmt = 5; 
% if fltr_rpt == 1
   FilteredImage=RefImage;
% end

for rpt= 1:lmt 

[FilteredImage] = AWRMedFilter(FilteredImage,BW,GrayImg); % FinalAdaptiveFilter(FilteredImage,BW,GrayImg,w);
end
InputImage = FilteredImage; 
% figure;
% imshow(mat2gray((FilteredImage)));
end

% %%$$$$$$$$$$$$$$$$$$$$$$$$$$$$%%
% InputImage = RefImage;
% [peaksnr, snr] = psnr(FilteredImage,InputImage)
% mse = immse(FilteredImage,InputImage)
% ssim = ssim(FilteredImage,RefImage)
% % ESSIM = ESSIM(InputImage,FilteredImage)
% % EPI = EPI(FilteredImage,InputImage)
% piqeO = piqe(InputImage)
% piqeF = piqe(FilteredImage)
% multissim = multissim(FilteredImage,InputImage)
% %%############################%%
% %%############################%%
% EI=[];EF=[];MI=[];JE=[];KLD=[]; JSD=[];
% for cb=1:3
% data_x = InputImage(:,:,cb); % data_x = data_x(1:m1*n1);
% data_y = FilteredImage(:,:,cb); % data_y = data_y(1:m2*n2);
% % N = 256;
% [p_x,p_y,p_xy] = jointhist(data_x,data_y); % p_xy = histcounts2(data_x,data_y,N); p_x = histcounts(data_x,N); p_y = histcounts(data_y,N);
% % sum(p_x)
% % sum(p_y)
% % sum(sum(p_xy)) % p_xy(p_xy == -inf & p_xy == -inf)=0; % p_xy = p_xy/sum(sum(p_xy));
% log2pxy = log2(p_xy);  log2pxy(log2pxy == -inf | log2pxy == inf)=0;
% % p_x(p_x == -inf & p_x == -inf)=0; % p_x = p_x/(sum(p_x));  
% % p_y(p_y == -inf & p_y == -inf)=0; % p_y = p_y/(sum(p_y));  
% log2px = log2(p_x); log2px(log2px == -inf | log2px == inf)=0;
% log2py = log2(p_y); log2py(log2py == -inf | log2py == inf)=0;
% px_y = p_x/p_y; px_y(px_y == -inf | px_y == inf)=0; log2px_y = log2(px_y);% px_y = px_y/(sum(px_y));
% py_x = p_y/p_x; py_x(py_x == -inf | py_x == inf)=0; log2py_x = log2(py_x); % py_x = py_x/(sum(py_x));
% log2px_y(log2px_y == -inf | log2px_y == inf)=0; log2py_x(log2py_x == -inf | log2py_x == inf)=0;
% p_x_y = (p_x+p_y)/2; p_x_yx = p_x_y/p_x; p_x_yy = p_x_y/p_y;
% p_x_yx(p_x_yx == -inf | p_x_yx == inf)=0; p_x_yy(p_x_yy == -inf | p_x_yy == inf)=0;
% log2p_x_yx = log2(p_x_yx); log2p_x_yx(log2p_x_yx == -inf | log2p_x_yx == inf)=0; % p_x_y = p_x_y/(sum(p_x_y)); 
% log2p_x_yy = log2(p_x_yy); log2p_x_yy(log2p_x_yy == -inf | log2p_x_yy == inf)=0;
% % p_x = p_x/sum(sum(p_x)); p_y = p_y/sum(sum(p_y)); 
% EI = [EI sum(-p_x.*log2px)];
% EF = [EF sum(-p_y.*log2py)];
% JE = [JE sum(sum(-p_xy.*log2pxy))];
% MI = [MI sum(-p_x.*log2px) + sum(-p_y.*log2py) - sum(sum(-p_xy.*log2pxy))];
% KLD = [KLD (sum(-p_x.*log2py_x) + sum(-p_y.*log2px_y))];
% JSD = [JSD (sum(-p_x.*log2p_x_yx) + sum(-p_y.*log2p_x_yy))/2];
% end
% MeanMI = mean(MI)
%   