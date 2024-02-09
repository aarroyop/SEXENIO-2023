%definitivo con crossvalidation
clear all;
close all;
 %fd = fopen('tr4.txt','r');%3295
%fd = fopen('tr3.txt','r');%3349
% fd = fopen('tr2.txt','r');%3453
%fd = fopen('tr1.txt','r'); %3429
 %fd = fopen('nourbanas.txt','r');%6763
 fd = fopen('boxplot.txt','r'); %6763
% fd = fopen('todas2.txt','r'); %13526
if (fd<0)
    'fopen failed'
    return;
end;
for (j=1:60)
     toma = fscanf(fd,'%s',1);
     season(j,1:length(toma)) = toma;
     
     neurons(j,1)=fscanf(fd,'%f',1);
     
     toma = fscanf(fd,'%s',1);
     algorithm(j,1:length(toma)) = toma;
     
     MSE_Mean(j,1)=fscanf(fd,'%f',1);
     MSE_STD(j,1)=fscanf(fd,'%f',1);
     MSE_Mean(j,1)=fscanf(fd,'%f',1);
     Time_STD(j,1)=fscanf(fd,'%f',1);
     j
    end;
    
%     x1 = normrnd(5,1,100,1);
% x2 = normrnd(6,1,100,1);
% x = randn(100,25);
% month = {'Jan' 'Feb' 'Mar' 'Apr'}';
% obsim = {'Obs' 'Sim'}';
% n = 400;
% boxplot(randn(n,1),{month(randi(4,n,1)),obsim(randi(2,n,1))},'factorsep',1,'factorgap',10)

boxplot(MSE_Mean,{ algorithm, neurons }, 'factorsep',1,'factorgap',10 )
 xlabel('#Neurons and Training algorithm')
 ylabel('MSE Mean')
 title('Season Dataset')
