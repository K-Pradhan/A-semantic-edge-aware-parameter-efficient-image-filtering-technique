function [gMean] = Gmean(x)
% x= [5 1 1 1 1]
% mean(x)
% 1:length(x)
gMean = sum(x.*exp(-(1:length(x)).^2)./(sum(exp(-(1:length(x)).^2))));
end