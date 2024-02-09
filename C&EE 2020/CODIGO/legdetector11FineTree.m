clear all; close all;
fd = fopen('legdetector.txt','r');
if (fd<0)
    'fopen failed';
    return;
end;
filas=21892; %18530
tramo1=2622;
tramo2=5980;
tramo3=21892;
columnas=9;
for (j=1:tramo3)     
        for d=1:columnas
         data1element = fscanf(fd,'%f',1);
         data(j,d) = data1element;
        end;
   
end;
data1=data(:,1:8);
data1=data1./norm(data1);
data1=[data1 data(:,9:9)]; %data1 tiene todas las filas normalizadas menos la última columna que se añade con 0 1
%filename='legdetectorhais.xlsx'
%xlswrite(filename,data1,'NORMALIZADO', 'A2');
data2=data1(1:tramo1,:); 
data21=data2; %tiene las 2622 primeras filas
data3=data1(tramo1+1:tramo2,:); %filas 2622 a 5980 que son las que vamos a imputar en función de las pocas primeras
for (i=7:8)
    data2=data21;
    data2(:,[i 1])=data21(:,[1 i]);
    data2(:,[1 i])=data21(:,[i 1]);
    Y=data2(:,1); %columna objetivo
    X=data2(:,2:9);
    
      XP=data3(:,2:9); %COLUMNA OBJETIVO DE LAS QUE VAMOS A PREDECIR
      
    %variables para TREE y BoostTree
    VarName1=data2(:,1); VarName2=data2(:,2); VarName3=data2(:,3); VarName4=data2(:,4); VarName5=data2(:,5); VarName6=data2(:,6); 
    VarName7=data2(:,7); VarName8=data2(:,8); VarName9=data2(:,9); 
    T = table(VarName1, VarName2, VarName3,VarName4,VarName5,VarName6,VarName7,VarName8,VarName9);
    
     %sacamos las 11 columnas de entrada para CValidation
    X1=X(:,1); X2=X(:,2); X3=X(:,3); X4=X(:,4); X5=X(:,5); X6=X(:,6); X7=X(:,7); X8=X(:,8);
     neurons=10; increment=4; %para RBF
%     tic
%       netR = newrb(X',Y', 60, 40,  neurons,increment) %creación de RBF para cada data1set %net = newrb(P,T,goal,spread,MN,DF) 
%     tocCreateRBF(i)=toc %lo que tarda en crearse RBF para cada data1set
    
%     modelfun = @(b,x)b(1) + b(2)*x(:,1).^b(3) + b(4)*x(:,2).^b(5)+b(6)*x(:,2); %para regresión no lineal multiple. estudiar si se puede sacar antes del bucle principal
%     beta0=[100 100 100 100 100 100];

    cv = cvpartition(length(Y),'Kfold',10);
    lx=1;   li=1;
    for j = 1:lx %diez veces para los 10 kfolds
       %común a perceptron, RB, regresión lineal y no lineal
       hiddenLayerSizeP = neurons;
       trainIdxs{j} = find(training(cv,j)); %90 para entrenamiento
       testIdxs{j} =  find(test(cv,j));   %10 para test
       trainMatrix{j} = [X1(trainIdxs{j}) X2(trainIdxs{j}) X3(trainIdxs{j}) X4(trainIdxs{j}) X5(trainIdxs{j}) X6(trainIdxs{j}) X7(trainIdxs{j}) X8(trainIdxs{j}) Y(trainIdxs{j})];
       testMatrix{j} = [X1(testIdxs{j}) X2(testIdxs{j}) X3(testIdxs{j}) X4(testIdxs{j}) X5(testIdxs{j}) X6(testIdxs{j}) X7(testIdxs{j}) X8(testIdxs{j}) Y(testIdxs{j})];         
%        %RNOLineal multiple 
%        tic
%           mdl = fitnlm(trainMatrix{j}(:,1:8),trainMatrix{j}(:,9),modelfun, beta0);
%           performanceRNoLineal(i,j) = mdl.MSE;
%        tocRNOlineal(i,j)=toc;
%        %RLineal multiple     
%        tic
%          Mdl = fitrlinear(trainMatrix{j}(:,1:8),trainMatrix{j}(:,9),'KFold',2,'Learner','leastsquares');
%          Mdl1 = Mdl.Trained{1};
%          performanceRlineal(i,j) = kfoldLoss(Mdl);
%        tocRLineal(i,j)=toc;
       %ejecucion de FineTree
       tic
         [trainedModel_FineTree, RMSE_FineTree(i,j)] = fineTree(T)
       tocFineTree(i, j)=toc;
       %ejecucion de boostedEnsemble
%        tic
%          [trainedModel_boostedEnsemble, RMSE_boostedEnsemble(i,j)] = boostedEnsemble(T)
%        tocboostedEnsemble(i, j)=toc;          
       %RBF
%        netR.divideFcn='divideind';
%        netR.divideParam.testInd = testIdxs{j};
%        netR.divideParam.trainInd = trainIdxs{j};
%        %netR.trainParam.epochs = 70;
%        tic
%           [yR] = netR(trainMatrix{j}(:,1:8)');
%           performanceRBF(i,j) = perform(netR,trainMatrix{j}(:,9)',yR);
%        tocRBFtrain(i,j)=toc;         
       %perceptrón MLP
%         for k=1:2
%           if (k==1) metodo='trainlm'; end;
%           if (k==2) metodo='trainbr'; end;     
%          tic 
%             trainFcnP = metodo;  %     
%             netP = fitnet(hiddenLayerSizeP,trainFcnP);
%           tocCreateMLP(i, j) =toc
%           netP.divideFcn='divideind';
%           netP.divideParam.testInd = testIdxs{j};
%           netP.divideParam.trainInd = trainIdxs{j};
%           netP.trainParam.epochs = 70;
%                    if k==1    
%               for l =1:li
%                   tic
%                    [netP, trP] = train(netP,trainMatrix{j}(:,1:8)',trainMatrix{j}(:,9)');
%                    [yP] = netP(trainMatrix{j}(:,1:8)');
%                    performancePLM(l) = perform(netP,trainMatrix{j}(:,9)',yP);
%                   tocPLMtrain(l)=toc; %en la versión del artículo aparecía tocPLMtrain(i)=toc; creo que es K
%                end;
%               MEDtocPLMtrain(i,j)=mean(tocPLMtrain);
%              STDtocPLMtrain(i,j)=std(tocPLMtrain);
%              medMSEPLM(i,j)=mean(performancePLM);
%              sMSEPLM(i,j)=std(performancePLM);
%           end;
%          if k==2
%               for l =1:li
%                 tic  
%                  [netP, trP] = train(netP,trainMatrix{j}(:,1:8)',trainMatrix{j}(:,9)');
%                 [yP] = netP(trainMatrix{j}(:,1:8)');
%                  performanceBR(l) = perform(netP,trainMatrix{j}(:,9)',yP);
%                 tocBRtrain(l)=toc;                 
%               end;
%              MEDtocPBRtrain(i,j)=mean(tocBRtrain);
%              STDtocPBRtrain(i,j)=std(tocBRtrain);
%               medMSEPBR(i,j)=mean(performanceBR)
%               sMSEPBR(i,j)=std(performanceBR);
%           end;  
%         end;        
  end;   
  data3(:,8)=0;
  data3(:,7)=0;
  if (i==7)
      
  VarName1=data3(:,7);   VarName2=data3(:,1);   VarName3=data3(:,2);   VarName4=data3(:,3);  VarName5=data3(:,4);    VarName6=data3(:,5);    VarName7=data3(:,6);    VarName8=data3(:,8);    VarName9=data3(:,9);
  T2=table(VarName1, VarName2, VarName3,VarName4,VarName5,VarName6,VarName7,VarName8,VarName9);
  end;
  if (i==8)
      
  VarName1=data3(:,8);   VarName2=data3(:,1);   VarName3=data3(:,2);   VarName4=data3(:,3);  VarName5=data3(:,4);    VarName6=data3(:,5);    VarName7=data3(:,6);    VarName8=data3(:,7);    VarName9=data3(:,9);
  T2=table(VarName1, VarName2, VarName3,VarName4,VarName5,VarName6,VarName7,VarName8,VarName9);
  end;
  for(h=1:3358)
%     y3RNOlineal(h,i-6)=predict(mdl,XP(h,:)); %
%     y3Rlineal(h,i-6)=predict(Mdl1,XP(h,:));

%     y3RBF(h,i-6)=sim(netR,XP(h,:)');
      y3FT(h,i-6)=trainedModel_FineTree.predictFcn(T2(h,:));
  
    
  end;
end;
%To make predictions on a new table, T, use:   
%yfit = c.predictFcn(T) replacing 'c' with the name of the variable that is this struct, e.g. 'trainedModel'.  
%The table, T, must contain the variables returned by:   c.RequiredVariables Variable formats (e.g. matrix/vector, datatype) must match the original training data. Additional variables are ignored.  
%For more information, see <a href="matlab:helpview(fullfile(docroot, 'stats', 'stats.map'), 'appregression_exportmodeltoworkspace')">How to predict using an exported model</a>.
filename='resultados11legdetectorFT.xlsx'; %solo se hace la regresión con el tramo1 y se estima en función de esas primeras filas con normalización

% xlswrite(filename,tocCreateRBF,'RBF', 'A2');
% xlswrite(filename,performanceRBF,'RBF','A4');
% xlswrite(filename,tocRBFtrain,'RBF','A19');
% 
% xlswrite(filename,y3RBF(1:3358,1),'imputacion','A3');
% xlswrite(filename,y3RBF(1:3358,2),'imputacion','B3');
% xlswrite(filename,y3RNOlineal(1:3358,1),'imputacion','D3');
% xlswrite(filename,y3RNOlineal(1:3358,2),'imputacion','E3');
% xlswrite(filename,y3Rlineal(1:3358,1),'imputacion','G3');
% xlswrite(filename,y3Rlineal(1:3358,2),'imputacion','H3');
% % 
%  xlswrite(filename,performanceRlineal,'R LINEAL','A3');
%  xlswrite(filename,tocRLineal,'R LINEAL','A18');
% 
% xlswrite(filename,performanceRNoLineal,'R NO LINEAL','A3');
% xlswrite(filename,tocRNOlineal,'R NO LINEAL','A18');

xlswrite(filename,RMSE_FineTree,'FINE TREE','A3');
xlswrite(filename,y3FT(1:3358,1),'imputacion','A3');
xlswrite(filename,y3FT(1:3358,2),'imputacion','B3');
%xlswrite(filename,tocFineTree,'FINE TREE','A18');
% 
% xlswrite(filename,RMSE_boostedEnsemble,'BOOSTED ENSEMBLE','A3');
% xlswrite(filename,tocboostedEnsemble,'BOOSTED ENSEMBLE','A18');
% 
% xlswrite(filename,tocCreateMLP,'MLP LM','A3');
% xlswrite(filename,MEDtocPLMtrain,'MLP LM','A18');
% xlswrite(filename,STDtocPLMtrain,'MLP LM','A33');
% xlswrite(filename,medMSEPLM,'MLP LM','A48');
% xlswrite(filename,sMSEPLM,'MLP LM','A63');
% 
% xlswrite(filename,tocCreateMLP,'MLP BR','A3');
% xlswrite(filename,MEDtocPBRtrain,'MLP BR','A18');
% xlswrite(filename,STDtocPBRtrain,'MLP BR','A33');
% xlswrite(filename,medMSEPBR,'MLP BR','A48');
% xlswrite(filename,sMSEPBR,'MLP BR','A63');





