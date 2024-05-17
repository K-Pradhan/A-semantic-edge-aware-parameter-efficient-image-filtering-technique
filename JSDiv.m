function [JSD] = JSDiv(LocalPattern,StandaredPattern)
% temp1=[]; temp2=[]; TotalPattern =[]; JSD=[];
if(nnz(LocalPattern)==0)
   LocalPattern = ones(1,length(LocalPattern));
end

StandaredPattern = StandaredPattern/sum(StandaredPattern);
LocalPattern = LocalPattern/sum(LocalPattern); 

TotalPattern = (LocalPattern + StandaredPattern)/2; 
LocalPattern = LocalPattern(TotalPattern~=0); 
StandaredPattern = StandaredPattern(TotalPattern~=0);
TotalPattern = TotalPattern(TotalPattern~=0);

tmp1 = LocalPattern ./ TotalPattern; 
tmp1 = log2(tmp1); 
tmp1(tmp1==-inf)=0; 
tmp2 = StandaredPattern ./ TotalPattern; 
tmp2 = log2(tmp2); 
tmp2(tmp2==-inf)=0;
JSD = (sum(LocalPattern .* tmp1) + sum(StandaredPattern .* tmp2))/2;

% BD = -log2(sum(sqrt(LocalPattern.*StandaredPattern)));
end