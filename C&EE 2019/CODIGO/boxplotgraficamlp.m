%definitivo con crossvalidation
clear all;
close all;
 %fd = fopen('tr4.txt','r');%3295
%fd = fopen('tr3.txt','r');%3349
% fd = fopen('tr2.txt','r');%3453
%fd = fopen('tr1.txt','r'); %3429
 %fd = fopen('nourbanas.txt','r');%6763
 fd = fopen('boxplot3.txt','r'); %6763
% fd = fopen('todas2.txt','r'); %13526
if (fd<0)
    'fopen failed'
    return;
end;
for (j=1:180)
%      toma = fscanf(fd,'%s',1);
%      Month(j,1:length(toma)) = toma;
     
%      toma = fscanf(fd,'%s',1);
%      Cluster(j,1:length(toma))=toma;
     
     toma = fscanf(fd,'%s',1);
     Algorithm(j,1:length(toma))=toma;
%      
     %algorithm(j,1:length(toma)) = toma;
     
     MSE_Mean(j,1)=fscanf(fd,'%f',1);
%       MSE_Mean_GDX(j,1)=fscanf(fd,'%f',1);
%       MSE_Mean_RB(j,1)=fscanf(fd,'%f',1);
%       MSE_Mean_SCG(j,1)=fscanf(fd,'%f',1);
%       MSE_Mean_BR(j,1)=fscanf(fd,'%f',1);
     j
    end;
    
%     x1 = normrnd(5,1,100,1);
% x2 = normrnd(6,1,100,1);
% x = randn(100,25);
% month = {'Jan' 'Feb' 'Mar' 'Apr'}';
% obsim = {'Obs' 'Sim'}';
% n = 400;
% boxplot(randn(n,1),{month(randi(4,n,1)),obsim(randi(2,n,1))},'factorsep',1,'factorgap',10)

boxplot(MSE_Mean,  {Algorithm}, 'factorsep',1,'factorgap',10, 'Whisker',10 )
%boxplot(MSE_Mean_LM, MSE_Mean_GDX, MSE_Mean_RB, MSE_Mean_SCG, MSE_Mean_BR, 'factorsep',1,'factorgap',5, 'Whisker',5 )
 xlabel('#Training algorithm')
 ylabel('MSE Mean')
 title('36-Grouped Dataset')
