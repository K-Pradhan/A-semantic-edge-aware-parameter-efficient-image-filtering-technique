function [FilteredImg] = AWRMedFilter(InputImage,BW,GrayImg)

[m , n, c] = size(InputImage); % output = InputImage;
FilteredImg = InputImage; % ZeroImg = zeros(m,n); 
if(c>1)
Inputbnd1 = InputImage(:,:,1);Inputbnd2 = InputImage(:,:,2);Inputbnd3 = InputImage(:,:,3);
else
Inputbnd1 = InputImage; Inputbnd2 = InputImage; Inputbnd3 = InputImage;
end

w1=1;
W_imdt = -w1:w1;
[X_imdt,Y_imdt] = meshgrid(-w1:w1,-w1:w1);
w4=2;
W_enbr = -w4:w4;
[X_enbr,Y_enbr] = meshgrid(-w4:w4,-w4:w4);
w2 = 2; %-floor(fltr_rpt/2);
W_edge = -w2:w2;
[X_edge,Y_edge] = meshgrid(-w2:w2,-w2:w2);
txt_end = 2;

parfor pxl = 1:m*n
    
% if(c>1)
% Inputbnd1 = InputImage(:,:,1);Inputbnd2 = InputImage(:,:,2);Inputbnd3 = InputImage(:,:,3);
% else
% Inputbnd1 = InputImage; Inputbnd2 = InputImage; Inputbnd3 = InputImage;
%  end 

[I,J] = ind2sub(size(GrayImg),pxl);
if (rem(pxl,m)==0)
    I = m; 
    J = fix(pxl/m);
else
    I = rem(pxl,m); 
    J = fix(pxl/m)+1;
end

% X1 = X_imdt+I; Y1 =Y_imdt+J;
% XY = (X1>0)&(X1<m) & (Y1>0)&(Y1<n);
% X1 = X1(XY);Y1 = Y1(XY);
% Nbr_imdt = sub2ind(size(GrayImg),X1,Y1);

Nbr_imdt = pxl + W_imdt*m; 
Nbr_imdt = repmat(Nbr_imdt,2*w1+1,1); 
Nbr_imdt = Nbr_imdt + Y_imdt; % ImdNbr = [ImdNbr - w ImdNbr + w];
Nbr_imdt = Nbr_imdt(Nbr_imdt>0 & Nbr_imdt<=m*n);

TxtrNbr = Nbr_imdt;

Nbr_enbr = pxl + W_enbr*m; 
Nbr_enbr = repmat(Nbr_enbr,2*w4+1,1); 
Nbr_enbr = Nbr_enbr + Y_enbr; % ImdNbr = [ImdNbr - w ImdNbr + w];
Nbr_enbr = Nbr_enbr(Nbr_enbr>0 & Nbr_enbr<=m*n);

% X1 = X+I; Y1 =Y+J;
% XY = (X1>0)&(X1<m) & (Y1>0)&(Y1<n);
% X1 = X1(XY);  Y1 = Y1(XY);
% % size(GrayImg)
% Nbr = sub2ind(size(GrayImg),X1,Y1);

Nbr = pxl + W_edge*m; 
Nbr = repmat(Nbr,2*w2+1,1); 
Nbr = Nbr + Y_edge; % ImdNbr = [ImdNbr - w ImdNbr + w];
%size(Nbr)
% SE = strel("disk",w2+1); %size(SE.Neighborhood)
% Nbr = Nbr.*SE.Neighborhood;
Nbr = Nbr(Nbr>0 & Nbr<=m*n);


if ( BW(pxl)==1) % && nnz(BW(Nbr_enbr))>=5 )  % BW(pxl)==1) %     nnz(BW(Nbr_enbr))>=3 ) %      || nnz(BW(Nbr_imdt)) > 0 ) % && skwns >= 0.30) %&& skwns/Range_Ratio >= 1.91) %  (mean(Range_Vals) - median(Range_Vals)  >=  std(Range_Vals)/2) % BW(pxl)==1 &&  (Pxl_Range(pxl) >= mean(Pxl_Range(Nbr))+std(Pxl_Range(Nbr)))
  
    midR = (max(GrayImg(Nbr)) + min(GrayImg(Nbr)))/2; 
     
     if ((median(GrayImg(Nbr_enbr)) <= midR)) 
         Nbr_New = Nbr(GrayImg(Nbr) <= midR);
         if(c>1)
         Filter_Wghts1 =  median(Inputbnd1(Nbr_New)); 
         Filter_Wghts2 =  median(Inputbnd2(Nbr_New));
         Filter_Wghts3 =  median(Inputbnd3(Nbr_New));
         else
         Filter_Wghts1 =  median(Inputbnd1(Nbr_New));
         end
     elseif((median(GrayImg(Nbr_enbr)) > midR))
         Nbr_New = Nbr(GrayImg(Nbr) > midR); 
         if(c>1)
         Filter_Wghts1 =  median(Inputbnd1(Nbr_New)); 
         Filter_Wghts2 =  median(Inputbnd2(Nbr_New));
         Filter_Wghts3 =  median(Inputbnd3(Nbr_New));
         else
         Filter_Wghts1 =  median(Inputbnd1(Nbr_New));
         end
     end
else
%     TxtrNbr_final = [];
    txtrwndw = txt_end;
    w3=txtrwndw;
%     TxtrNbr{txtrwndw}
%     BW(TxtrNbr{txtrwndw})
%     nnz(BW(TxtrNbr));
    boundaryEntry = 0;
    while (nnz(BW(TxtrNbr))<= boundaryEntry+1 && w3<=34) % 2*txtrwndw+1
          boundaryEntry = nnz(BW(TxtrNbr));
          txtrwndw = txtrwndw + 1;
          w3=txtrwndw;
          [X_texture,Y_texture] = meshgrid(-w3:w3,-w3:w3);
          X1 = X_texture+I; Y1 = Y_texture+J;
%           SE = strel("disk",w3+1); X1 = X1.*SE.Neighborhood; Y1 = Y1.*SE.Neighborhood;
          XY = (X1>0)&(X1<=m) & (Y1>0)&(Y1<=n);
          X1 = X1(XY);
          Y1 = Y1(XY);
          TxtrNbr = sub2ind([m,n],X1,Y1);
    end

    w3=txtrwndw-1; % halt;
    [X_texture,Y_texture] = meshgrid(-w3:w3,-w3:w3);
    X1 = X_texture+I; Y1 = Y_texture+J; %size(X1)
%     SE = strel("disk",w3+1); % size(SE.Neighborhood)
%     X1 = X1.*SE.Neighborhood; Y1 = Y1.*SE.Neighborhood;
    XY = (X1>0)&(X1<=m) & (Y1>0)&(Y1<=n);
    X1 = X1(XY); Y1 = Y1(XY);
    TxtrNbr_final = sub2ind([m,n],X1,Y1);
%     TxtrNbr_final = TxtrNbr;

    
    if(c>1)
       Filter_Wghts1 =  median(Inputbnd1(TxtrNbr_final));
       Filter_Wghts2 =  median(Inputbnd2(TxtrNbr_final));
       Filter_Wghts3 =  median(Inputbnd3(TxtrNbr_final));
    else
       Filter_Wghts1 =  median(Inputbnd1(TxtrNbr_final));
    end

end

%     FilteredImg(pxl) =  Filter_Wghts1;
    if(c>1)
        output(pxl,:) = [Filter_Wghts1; Filter_Wghts2; Filter_Wghts3];
%         output = reshape(output,[m n c]);
%     Inputbnd1(pxl,:) = [Filter_Wghts1; Filter_Wghts2; Filter_Wghts3]; % = reshape(output,m,n,[]);
    else
    output(pxl,:) =  [Filter_Wghts1; Filter_Wghts1; Filter_Wghts1];
%     Inputbnd1(pxl) = Filter_Wghts1;Inputbnd2(pxl) = Filter_Wghts1;Inputbnd3(pxl) = Filter_Wghts1; %reshape(output(:,1),m,[]);
    end
    
%     if(c>1)
%      Inputbnd1 = output(:,1); Inputbnd2 = output(:,2); Inputbnd3 = output(:,3);
%     else
%      Inputbnd1 = output; Inputbnd2 = output; Inputbnd3 = output;
%     end
%     InputImage = reshape(output,m,n,[]);
end

%  figure;
%  imshow(mat2gray(FilteredImg));  % halt;
 if(c>1) 
     FilteredImg = reshape(output,[m n c]); % FilteredImg =  output; % 
 else
    FilteredImg = reshape(output(:,1),[m n 1]);
 end
 disp('Time for Adaptive Filtering Restoration =');
%  toc;
 end
