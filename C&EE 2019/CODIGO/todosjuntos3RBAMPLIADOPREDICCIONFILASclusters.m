%definitivo con crossvalidation para doce meses y 3 clusters mes. guarda
%todos los datos en un solo ficheros y leemos de cualquier fichero con
%strcat
clear all; close all;

for (z=1:12) %meses
%       clear all;
%         close all;
    for (w= 1:3) %clusters por mes
      close all;
      clearvars -except z w
 zz=int2str(z); ww =int2str(w); entrada=strcat('Mes',zz,'cluster',ww,'.txt');    
      filename='kresultadoglobal4.xlsx';%342
 fd = fopen(entrada,'r');%6763
if (fd<0)
    'fopen failed'
    return;
end;
datos=1;
dataelement(1,:) = fscanf(fd,'%f',7);
while ~feof(fd)
     x = fscanf(fd,'%f',7);
     if (x >-100) 
         datos=datos+1;
         dataelement(datos,:) = x; 
     end;
     
  end;

dataelement=dataelement./norm(dataelement);
Y=dataelement(:,1); %columna objetivo
X=dataelement(:,2:7);

%para cargar nuevas filas y predecir su valor
% for (j=1:50)
%      Yp(j,1) = fscanf(fd,'%f',1);
%     for d=1:6
%         dataelement = fscanf(fd,'%f',1);
%         Xp(j,d) = dataelement;
%     end;
% end;
 %R = corrcoef(prueba);%coeficiente de correlaci�n
neurons=10;
increment=1;
 %sacamos las 5 columnas de entrada para CValidation
 X1=X(:,1); X2=X(:,2); X3=X(:,3); X4=X(:,4); X5=X(:,5); X6=X(:,6);
%para RBF
tic
%   net = newrb(P,T,goal,spread,MN,DF) 
  netR = newrb(X',Y', 100, 40,  neurons,increment)
tocCreateRB=toc
%para regresi�n no lineal multiple
modelfun = @(b,x)b(1) + b(2)*x(:,1).^b(3) + b(4)*x(:,2).^b(5)+b(6)*x(:,2);
%beta0=[0 0 0 0 0];
beta0=[100 100 100 100 100 100];

cv = cvpartition(length(Y),'Kfold',10);
%[trainInd,valInd,testInd] = dividerand(1400,0.7,0.15,0.15)
lx=10;
li=10;
%sumMSERLineal=0; sumMSERNOLineal=0; %SOBRAR�A ESTA Y LAS DE ABAJO SI HACE BIEN LA MEDIA DE LOS
     %VECTORES
for i = 1:lx %diez veces para los 10 kfolds
     %com�n a perceptron, RB, regresi�n lineal y no lineal
     hiddenLayerSizeP = neurons;
     trainIdxs{i} = find(training(cv,i)); %90 para entrenamiento
     testIdxs{i} =  find(test(cv,i));   %10 para test
     trainMatrix{i} = [X1(trainIdxs{i}) X2(trainIdxs{i}) X3(trainIdxs{i}) X4(trainIdxs{i}) X5(trainIdxs{i}) X6(trainIdxs{i}) Y(trainIdxs{i})];
     testMatrix{i} = [X1(testIdxs{i}) X2(testIdxs{i}) X3(testIdxs{i}) X4(testIdxs{i}) X5(testIdxs{i}) X6(testIdxs{i}) Y(testIdxs{i})];    
     datos3=size(testMatrix{i})
     datos2=size(trainMatrix{i})
     %RNOLineal multiple
     
      mdl = fitnlm(trainMatrix{i}(:,1:6),trainMatrix{i}(:,7),modelfun, beta0);
      performanceRNoLineal(i) = mdl.MSE;
     tocRNOlineal(i)=toc  
           
     %RBF
     netR.divideFcn='divideind';
     netR.divideParam.testInd = testIdxs{i};
     netR.divideParam.trainInd = trainIdxs{i};
          %netR.trainParam.epochs = 70;
           
                 tic
                  %[netR, trR] = train(netR,trainMatrix{i}(:,1:6)',trainMatrix{i}(:,7)');
                  [yR] = netR(trainMatrix{i}(:,1:6)');
                    %eR = gsubtract(trainMatrix{i}(:,6)',yR);
                  performanceR(i) = perform(netR,trainMatrix{i}(:,7)',yR);
                 tocRtrain(i)=toc; %en la versi�n del art�culo aparec�a tocPLMtrain(i)=toc; creo que es K                        
     %Gr�ficas de regresi�n para RBF            
%      [rRLM(i),mRLM(i),bRLM(i)] = regression(trainMatrix{i}(:,7)',yR);% ESTA LINEA NO SIRVE
%      figure(), fig=plotregression(trainMatrix{i}(:,7)',yR); %REVISAR Y MEJORAR LA GR�FICA DE REGRESI�N
%      if (i==1) saveas(fig,'1kfoldRBF.fig');  saveas(fig,'1kfoldRBF.tiff'); end;
%      if (i==2) saveas(fig,'2kfoldRBF.fig');  saveas(fig,'2kfoldRBF.tiff'); end;
%      if (i==3) saveas(fig,'3kfoldRBF.fig');  saveas(fig,'3kfoldRBF.tiff'); end;
%      if (i==4) saveas(fig,'4kfoldRBF.fig');  saveas(fig,'4kfoldRBF.tiff'); end;
%      if (i==5) saveas(fig,'5kfoldRBF.fig');  saveas(fig,'5kfoldRBF.tiff'); end;
%      if (i==6) saveas(fig,'6kfoldRBF.fig');  saveas(fig,'6kfoldRBF.tiff'); end;
%      if (i==7) saveas(fig,'7kfoldRBF.fig');  saveas(fig,'7kfoldRBF.tiff'); end;
%      if (i==8) saveas(fig,'8kfoldRBF.fig');  saveas(fig,'8kfoldRBF.tiff'); end;
%      if (i==9) saveas(fig,'9kfoldRBF.fig');  saveas(fig,'9kfoldRBF.tiff'); end;
%      if (i==10) saveas(fig,'10kfoldRBF.fig');  saveas(fig,'10kfoldRBF.tiff'); end;
%      close(fig);
     %perceptr�n
        for j=1:5
         if (j==1) metodo='trainlm'; end;
         if (j==2) metodo='traingdx'; end; 
         if (j==3) metodo='trainb'; end;
         if (j==4) metodo='trainscg'; end;
         if (j==5) metodo='trainbr'; end;     
      
     %PERCEPTRON MLP
        tic 
           trainFcnP = metodo;  % Levenberg-Marquardt
           
           netP = fitnet(hiddenLayerSizeP,trainFcnP);
         tocCreatePER(j) =toc
         netP.divideFcn='divideind';
         netP.divideParam.testInd = testIdxs{i};
         netP.divideParam.trainInd = trainIdxs{i};
         netP.trainParam.epochs = 70;
         if j==1
             
             for k =1:li
                 tic
                  [netP, trP] = train(netP,trainMatrix{i}(:,1:6)',trainMatrix{i}(:,7)');
                  [yP] = netP(trainMatrix{i}(:,1:6)');
                    %eR = gsubtract(trainMatrix{i}(:,6)',yR);
                  performancePLM(k) = perform(netP,trainMatrix{i}(:,7)',yP);
                 tocPLMtrain(k)=toc; %en la versi�n del art�culo aparec�a tocPLMtrain(i)=toc; creo que es K
             
                %[rRLM(k),mRLM(i),bRLM(i)] = regression(trainMatrix{i}(:,6)',yR);  
             end;
             MEDtocPLMtrain(i)=median(tocPLMtrain);
             STDtocPLMtrain(i)=std(tocPLMtrain);
             medMSEPLM(i)=median(performancePLM);
             sMSEPLM(i)=std(performancePLM);
         end;
         if j==2
            
            for k =1:li
              tic  
                [netP, trP] = train(netP,trainMatrix{i}(:,1:6)',trainMatrix{i}(:,7)');
                [yP] = netP(trainMatrix{i}(:,1:6)');
                performancePGDX(k) = perform(netP,trainMatrix{i}(:,7)',yP);
              tocPGDXtrain(k)=toc;  
                
            end;
            MEDtocPGDXtrain(i)=median(tocPGDXtrain);
             STDtocPGDXtrain(i)=std(tocPGDXtrain);
            medMSEPGDX(i)=median(performancePGDX);
            sMSEPGDX(i)=std(performancePGDX);
         end;
         if j==3
            for k =1:li
              tic   
                [netP, trP] = train(netP,trainMatrix{i}(:,1:6)',trainMatrix{i}(:,7)');
                [yP] = netP(trainMatrix{i}(:,1:6)');
                performancePB(k) = perform(netP,trainMatrix{i}(:,7)',yP);
               tocPBtrain(k)=toc;
                
             end;
            MEDtocPBtrain(i)=median(tocPBtrain);
             STDtocPBtrain(i)=std(tocPBtrain);
             medMSEPB(i)=median(performancePB);
             sMSEPB(i)=std(performancePB);
         end;
         if j==4
             
             for k =1:li
               tic  
                [netP, trP] = train(netP,trainMatrix{i}(:,1:6)',trainMatrix{i}(:,7)');
                [yP] = netP(trainMatrix{i}(:,1:6)');
                performancePSCG(k) = perform(netP,trainMatrix{i}(:,7)',yP);
               tocPSCGtrain(k)=toc; 
                  
             end;
             MEDtocPSCGtrain(i)=median(tocPSCGtrain);
             STDtocPSCGtrain(i)=std(tocPSCGtrain);
             medMSEPSCG(i)=median(performancePSCG);
             sMSEPSCG(i)=std(performancePSCG);
         end;
         if j==5
             
             for k =1:li
               tic  
                [netP, trP] = train(netP,trainMatrix{i}(:,1:6)',trainMatrix{i}(:,7)');
                [yP] = netP(trainMatrix{i}(:,1:6)');
                performancePBR(k) = perform(netP,trainMatrix{i}(:,7)',yP);
               tocPBRtrain(k)=toc; 
                
             end;
             MEDtocPBRtrain(i)=median(tocPBRtrain);
             STDtocPBRtrain(i)=std(tocPBRtrain);
             medMSEPBR(i)=median(performancePBR)
             sMSEPBR(i)=std(performancePBR);
          end;  
         end;
     
    %RLineal multiple     
     tic
      Mdl = fitrlinear(trainMatrix{i}(:,1:6),trainMatrix{i}(:,7),'KFold',2,'Learner','leastsquares');
      Mdl1 = Mdl.Trained{1};
      performanceRlineal(i) = kfoldLoss(Mdl);
     tocRLineal(i)=toc
     

end;
%predicci�n para los 4 m�todos de las 50 nuevas filas
% for(h=1:50)
%     y3Rlineal(h)=predict(Mdl1,Xp(h,:));
%     y3RNOlineal(h)=predict(mdl,Xp(h,:));
%     y3RBF(h)=sim(netR,Xp(h,:)');
%     y3MLP(h)=sim(netP,Xp(h,:)');
%     
% end;
%RESULTADOS
%RB
%RESULTADOS VIEJOS DEL ARTICULO DE HINDAWI PARA RB
TmedMSER=mean(performanceR); %media de los errores. POSIBLE ERROR SI NO HACE LA MEDIA DEL VECTOR
TsSMER=std(performanceR); %DESVIACI�N DE LOS ERORES. POSIBLE ERROR SI NO HACE LA MEDIA DEL VECTOR
TmedR=mean(tocRtrain); %media de los TIEMPOS. POSIBLE ERROR SI NO HACE LA MEDIA DEL VECTOR
STDmedR=std(tocRtrain);
% tocRLMtrainMED=mean(MEDtocRLMtrain); %media de la media de los tiempos de entreno, 10 vueltas para cada uno de los 10 kfolds
% tocRGDXtrainMED=mean(MEDtocRGDXtrain);
% tocRBtrainMED=mean(MEDtocRBtrain);
% tocRSCGtrainMED=mean(MEDtocRSCGtrain);
% tocRBRtrainMED=mean(MEDtocRBRtrain);
% 
% tocRLMtrainSTD=std(STDtocRLMtrain); %desviaci�n t�poca de la desviaci�n t�pica
% tocRGDXtrainSTD=std(STDtocRGDXtrain);
% tocRBtrainSTD=std(STDtocRBtrain);
% tocRSCGtrainSTD=std(STDtocRSCGtrain);
% tocRBRtrainSTD=std(STDtocRBRtrain);
% 
% TmedMSERLM=mean(medMSERLM);
% TmedMSERGDX=mean(medMSERGDX);
% TmedMSERB=mean(medMSERB);
% TmedMSERSCG=mean(medMSERSCG);
% TmedMSERBR=mean(medMSERBR);
% 
% TssmeRLM=std(medMSERLM);
% TssmeRGDX=std(medMSERGDX);
% TssmeRB=std(medMSERB);
% TssmeRSCG=std(medMSERSCG);
% TssmeRBR=std(medMSERBR);


% perceptron
tocCreatePER = mean(tocCreatePER); %MEDIA DE TIEMPOS EN ENTRENAR LAS REDES PARA LOS DISTINTOS METODOS, NO TIENE SENTIDO

tocPLMtrainMED=mean(MEDtocPLMtrain); %media de la media de los tiempos de entreno, 10 vueltas para cada uno de los 10 kfolds
tocPGDXtrainMED=mean(MEDtocPGDXtrain);
tocPBtrainMED=mean(MEDtocPBtrain);
tocPSCGtrainMED=mean(MEDtocPSCGtrain);
tocPBRtrainMED=mean(MEDtocPBRtrain);

tocPLMtrainSTD=std(STDtocPLMtrain); %desviaci�n t�poca de la desviaci�n t�pica
tocPGDXtrainSTD=std(STDtocPGDXtrain);
tocPBtrainSTD=std(STDtocPBtrain);
tocPSCGtrainSTD=std(STDtocPSCGtrain);
tocPBRtrainSTD=std(STDtocPBRtrain);

TmedMSEPLM=mean(medMSEPLM);
TmedMSEPGDX=mean(medMSEPGDX);
TmedMSEPB=mean(medMSEPB);
TmedMSEPSCG=mean(medMSEPSCG);
TmedMSEPBR=mean(medMSEPBR);

TssmePLM=std(medMSEPLM);
TssmePGDX=std(medMSEPGDX);
TssmePB=std(medMSEPB);
TssmePSCG=std(medMSEPSCG);
TssmePBR=std(medMSEPBR);

%rlineal
MedRLineal=mean(performanceRlineal); %media de las performance
ssmeRLineal=std(performanceRlineal);%desviaci�n de las performance. POSIBLE ERROR, VER SI HACE LA MEDIA DE LOS 10 VALORES DEL VECTO
tocRLinealmed=mean(tocRLineal);%media de los tiempos de ejecuci�n. POSIBLE ERROR, VER SI HACE LA MEDIA DE LOS 10 VALORES DEL VECTO
tocRLinealstd=std(tocRLineal);%desviaci�n de los tiempos de ejecuci�n. POSIBLE ERROR, VER SI HACE LA STD DE LOS 10 VALORES DEL VECTO
% %RNOLineal
 MedRnoLineal=mean(performanceRNoLineal);
ssmeRNOLineal=std(performanceRNoLineal);
tocRNOlinealmed=mean(tocRNOlineal);
tocRNOlinealstd=std(tocRNOlineal)

if (z==1 && w==1) A = {'filastrain','filastest','month','cluster','Time RBF','MSE Mean','MSE STD', 'T med R','STD R','Neurons';datos2(1),datos3(1),z,w,tocCreateRB,TmedMSER,TsSMER,TmedR,STDmedR,neurons}; end;

if (z>1 || w>1) A= {datos2(1),datos3(1),z,w,tocCreateRB,TmedMSER,TsSMER,TmedR,STDmedR,neurons}; end;
%A = {'filas','cluster','Time RBF','MSE Mean','MSE STD', 'T med R','STD R','Neurons';datos,z,tocCreateRB,TmedMSER,TsSMER,TmedR,STDmedR,neurons};%resultados viejos rb
%A2 MOSTRAR�A LOS DATOS INDIVIDUALES DE CADA KFOLD = {'K-Fold','MSE';1,performanceR(1);2,performanceR(2);3,performanceR(3);4,performanceR(4);5,performanceR(5);6,performanceR(6);7,performanceR(7);8,performanceR(8);9,performanceR(9);10,performanceR(10)};
%A3 = {'STD LM','STD GDX','STD RB','STD SCG','STD BR';sMSERLM(1),sMSERGDX(1),sMSERB(1),sMSERSCG(1),sMSERBR(1);sMSERLM(2),sMSERGDX(2),sMSERB(2),sMSERSCG(2),sMSERBR(2)}
% A = {'Time Rb','Neurons';tocCreateRB,hiddenLayerSizeP};
% A2={'MSE Mean LM','MSE Mean GDX','MSE Mean RB','MSE Mean SCG','MSE Mean BR';TmedMSERLM,TmedMSERGDX,TmedMSERB,TmedMSERSCG,TmedMSERBR};
% A3={'STD LM','MSE STD GDX','MSE STD RB','MSE STD SCG','MSE STD BR';TssmeRLM,TssmeRGDX,TssmeRB,TssmeRSCG,TssmeRBR};
% A8={'Time Mean LM','Time Mean GDX','Time Mean RB','Time Mean SCG','Time Mean BR';tocRLMtrainMED,tocRGDXtrainMED,tocRBtrainMED,tocRSCGtrainMED,tocRBRtrainMED};
% A9={'Time Std LM','Time Std GDX','Time Std RB','Time Std SCG','Time Std BR';tocRLMtrainSTD,tocRGDXtrainSTD,tocRBtrainSTD,tocRSCGtrainSTD,tocRBRtrainSTD};

%B4, B5, B6 Y B7 Nno se usan en el art�culo, son los tiempos medios de cada
%kfold
%B4={'K-Fold','MSE Mean LM','MSE Mean GDX','MSE Mean RB','MSE Mean SCG','MSE Mean BR';1,medMSEPLM(1),medMSEPGDX(1),medMSEPB(1),medMSEPSCG(1),medMSEPBR(1);2,medMSEPLM(2),medMSEPGDX(2),medMSEPB(2),medMSEPSCG(2),medMSEPBR(2);3,medMSEPLM(3),medMSEPGDX(3),medMSEPB(3),medMSEPSCG(3),medMSEPBR(3);4,medMSEPLM(4),medMSEPGDX(4),medMSEPB(4),medMSEPSCG(4),medMSEPBR(4);5,medMSEPLM(5),medMSEPGDX(5),medMSEPB(5),medMSEPSCG(5),medMSEPBR(5);6,medMSEPLM(6),medMSEPGDX(6),medMSEPB(6),medMSEPSCG(6),medMSEPBR(6);7,medMSEPLM(7),medMSEPGDX(7),medMSEPB(7),medMSEPSCG(7),medMSEPBR(7);8,medMSEPLM(8),medMSEPGDX(8),medMSEPB(8),medMSEPSCG(8),medMSEPBR(8);9,medMSEPLM(9),medMSEPGDX(9),medMSEPB(9),medMSEPSCG(9),medMSEPBR(9);10,medMSEPLM(10),medMSEPGDX(10),medMSEPB(10),medMSEPSCG(10),medMSEPBR(10)};
%B5={'K-Fold','MSE STD LM','MSE STD GDX','MSE STD RB','MSE STD SCG','MSE STD BR';1,sMSEPLM(1),sMSEPGDX(1),sMSEPB(1),sMSEPSCG(1),sMSEPBR(1);2,sMSEPLM(2),sMSEPGDX(2),sMSEPB(2),sMSEPSCG(2),sMSEPBR(2);3,sMSEPLM(3),sMSEPGDX(3),sMSEPB(3),sMSEPSCG(3),sMSEPBR(3);4,sMSEPLM(4),sMSEPGDX(4),sMSEPB(4),sMSEPSCG(4),sMSEPBR(4);5,sMSEPLM(5),sMSEPGDX(5),sMSEPB(5),sMSEPSCG(5),sMSEPBR(5);6,sMSEPLM(6),sMSEPGDX(6),sMSEPB(6),sMSEPSCG(6),sMSEPBR(6);7,sMSEPLM(7),sMSEPGDX(7),sMSEPB(7),sMSEPSCG(7),sMSEPBR(7);8,sMSEPLM(8),sMSEPGDX(8),sMSEPB(8),sMSEPSCG(8),sMSEPBR(8);9,sMSEPLM(9),sMSEPGDX(9),sMSEPB(9),sMSEPSCG(9),sMSEPBR(9);10,sMSEPLM(10),sMSEPGDX(10),sMSEPB(10),sMSEPSCG(10),sMSEPBR(10)};
%B6={'K-Fold','Time Mean LM','Time Mean GDX','Time Mean RB','Time Mean SCG','Time Mean BR';1,MEDtocPLMtrain(1),MEDtocPGDXtrain(1),MEDtocPBtrain(1),MEDtocPSCGtrain(1),MEDtocPBRtrain(1);2,MEDtocPLMtrain(2),MEDtocPGDXtrain(2),MEDtocPBtrain(2),MEDtocPSCGtrain(2),MEDtocPBRtrain(2);3,MEDtocPLMtrain(3),MEDtocPGDXtrain(3),MEDtocPBtrain(3),MEDtocPSCGtrain(3),MEDtocPBRtrain(3);4,MEDtocPLMtrain(4),MEDtocPGDXtrain(4),MEDtocPBtrain(4),MEDtocPSCGtrain(4),MEDtocPBRtrain(4);5,MEDtocPLMtrain(5),MEDtocPGDXtrain(5),MEDtocPBtrain(5),MEDtocPSCGtrain(5),MEDtocPBRtrain(5);6,MEDtocPLMtrain(6),MEDtocPGDXtrain(6),MEDtocPBtrain(6),MEDtocPSCGtrain(6),MEDtocPBRtrain(6);7,MEDtocPLMtrain(7),MEDtocPGDXtrain(7),MEDtocPBtrain(7),MEDtocPSCGtrain(7),MEDtocPBRtrain(7);8,MEDtocPLMtrain(8),MEDtocPGDXtrain(8),MEDtocPBtrain(8),MEDtocPSCGtrain(8),MEDtocPBRtrain(8);9,MEDtocPLMtrain(9),MEDtocPGDXtrain(9),MEDtocPBtrain(9),MEDtocPSCGtrain(9),MEDtocPBRtrain(9);10,MEDtocPLMtrain(10),MEDtocPGDXtrain(10),MEDtocPBtrain(10),MEDtocPSCGtrain(10),MEDtocPBRtrain(10)};
%B7={'K-Fold','Time Std LM','Time Std GDX','Time Std RB','Time Std SCG','Time Std BR';1,STDtocPLMtrain(1),STDtocPGDXtrain(1),STDtocPBtrain(1),STDtocPSCGtrain(1),STDtocPBRtrain(1);2,STDtocPLMtrain(2),STDtocPGDXtrain(2),STDtocPBtrain(2),STDtocPSCGtrain(2),STDtocPBRtrain(2);3,STDtocPLMtrain(3),STDtocPGDXtrain(3),STDtocPBtrain(3),STDtocPSCGtrain(3),STDtocPBRtrain(3);4,STDtocPLMtrain(4),STDtocPGDXtrain(4),STDtocPBtrain(4),STDtocPSCGtrain(4),STDtocPBRtrain(4);5,STDtocPLMtrain(5),STDtocPGDXtrain(5),STDtocPBtrain(5),STDtocPSCGtrain(5),STDtocPBRtrain(5);6,STDtocPLMtrain(6),STDtocPGDXtrain(6),STDtocPBtrain(6),STDtocPSCGtrain(6),STDtocPBRtrain(6);7,STDtocPLMtrain(7),STDtocPGDXtrain(7),STDtocPBtrain(7),STDtocPSCGtrain(7),STDtocPBRtrain(7);8,STDtocPLMtrain(8),STDtocPGDXtrain(8),STDtocPBtrain(8),STDtocPSCGtrain(8),STDtocPBRtrain(8);9,STDtocPLMtrain(9),STDtocPGDXtrain(9),STDtocPBtrain(9),STDtocPSCGtrain(9),STDtocPBRtrain(9);10,STDtocPLMtrain(10),STDtocPGDXtrain(10),STDtocPBtrain(10),STDtocPSCGtrain(10),STDtocPBRtrain(10)};
if (z==1 && w==1) B = {'month','cluster', 'Time Perceptron','Neurons';z,w,tocCreatePER,hiddenLayerSizeP}; end;
if (z>1 || w>1) B= {z,w,tocCreatePER,hiddenLayerSizeP}; end;
if (z==1 && w==1) B2={'month','cluster','MSE Mean LM','MSE Mean GDX','MSE Mean RB','MSE Mean SCG','MSE Mean BR';z,w,TmedMSEPLM,TmedMSEPGDX,TmedMSEPB,TmedMSEPSCG,TmedMSEPBR}; end;
if ( z>1 || w>1) B2={z,w,TmedMSEPLM,TmedMSEPGDX,TmedMSEPB,TmedMSEPSCG,TmedMSEPBR}; end;
if ( z==1 && w==1) B3={'month','cluster','STD LM','MSE STD GDX','MSE STD RB','MSE STD SCG','MSE STD BR';z,w,TssmePLM,TssmePGDX,TssmePB,TssmePSCG,TssmePBR}; end;
if ( z>1 || w>1) B3={z,w,TssmePLM,TssmePGDX,TssmePB,TssmePSCG,TssmePBR}; end;
if (z==1 && w==1) B8={'month','cluster','Time Mean LM','Time Mean GDX','Time Mean RB','Time Mean SCG','Time Mean BR';z,w,tocPLMtrainMED,tocPGDXtrainMED,tocPBtrainMED,tocPSCGtrainMED,tocPBRtrainMED}; end;
if (z>1 || w>1) B8={z,w,tocPLMtrainMED,tocPGDXtrainMED,tocPBtrainMED,tocPSCGtrainMED,tocPBRtrainMED}; end;
if (z==1 && w==1) B9={'month','cluster','Time Std LM','Time Std GDX','Time Std RB','Time Std SCG','Time Std BR';z,w,tocPLMtrainSTD,tocPGDXtrainSTD,tocPBtrainSTD,tocPSCGtrainSTD,tocPBRtrainSTD}; end;
if ( z>1 || w>1) B9={z,w,tocPLMtrainSTD,tocPGDXtrainSTD,tocPBtrainSTD,tocPSCGtrainSTD,tocPBRtrainSTD}; end;

if ( z==1 && w==1) C = {'month','cluster','Time R Lineal','Std Time R NO Lineal','MSE Mean','MSE STD';z,w,tocRLinealmed,tocRLinealstd,MedRLineal,ssmeRLineal}; end;
if (z>1 || w>1) C={z,w,tocRLinealmed,tocRLinealstd,MedRLineal,ssmeRLineal}; end;
%MOSTRAR�AMOS LOS DATOS DE CADA KFOLD C2={'K-Fold','MSE';1,performanceRlineal(1);2,performanceRlineal(2);3,performanceRlineal(3);4,performanceRlineal(4);5,performanceRlineal(5);6,performanceRlineal(6);7,performanceRlineal(7);8,performanceRlineal(8);9,performanceRlineal(9);10,performanceRlineal(10)};
if ( z==1 && w==1) D = {'month','cluster','Time R NO Lineal','Std Time R NO Lineal','MSE Mean','MSE STD';z,w,tocRNOlinealmed,tocRNOlinealstd,MedRnoLineal,ssmeRNOLineal}; end;
if ( z>1 || w>1) D = {z,w,tocRNOlinealmed,tocRNOlinealstd,MedRnoLineal,ssmeRNOLineal}; end;
 %MOSTRAR�AMOS LOS DATOS DE CADA KFOLD D2={'K-Fold','MSE';1,performanceRNoLineal(1);2,performanceRNoLineal(2);3,performanceRNoLineal(3);4,performanceRNoLineal(4);5,performanceRNoLineal(5);6,performanceRNoLineal(6);7,performanceRNoLineal(7);8,performanceRNoLineal(8);9,performanceRNoLineal(9);10,performanceRNoLineal(10)};

if (z==1 && w==1) xlswrite(filename,A,'RBF','A1'); end; 
f=(z*3+w)-2; f=int2str(f); f=strcat('A',f);
if (z>1 || w>1) xlswrite(filename,A,'RBF',f); end; 

% xlswrite(filename,A2,'RBF','A5'); 
% xlswrite(filename,A3,'RBF','A10');
% xlswrite(filename,A8,'RBF','A20');
% xlswrite(filename,A9,'RBF','A25');
if ( z==1 && w==1) xlswrite(filename,B,'Perceptron','A1'); end;
f=(z*3+w)-2; f=int2str(f); f=strcat('A',f);
if ( z>1 || w>1) xlswrite(filename,B,'Perceptron',f); end;

if ( z==1 && w==1) xlswrite(filename,B2,'Perceptron','A40'); end;
f=(z*3+w)+37; f=int2str(f); f=strcat('A',f);
if ( z>1 || w>1) xlswrite(filename,B2,'Perceptron',f); end;

if ( z==1 && w==1) xlswrite(filename,B3,'Perceptron','A80'); end;
f=(z*3+w)+77; f=int2str(f); f=strcat('A',f);
if (z>1 || w>1) xlswrite(filename,B3,'Perceptron',f); end;

% xlswrite(filename,B4,'Perceptron','A10');
% xlswrite(filename,B5,'Perceptron','A25');
% xlswrite(filename,B6,'Perceptron','A40');
% xlswrite(filename,B7,'Perceptron','A55');
if (  z==1 && w==1) xlswrite(filename,B8,'Perceptron','A120'); end;
f=(z*3+w)+117; f=int2str(f); f=strcat('A',f);
if ( z>1 || w>1) xlswrite(filename,B8,'Perceptron',f); end;

if ( z==1 && w==1) xlswrite(filename,B9,'Perceptron','A160'); end;
f=(z*3+w)+157; f=int2str(f); f=strcat('A',f);
if ( z>1 || w>1) xlswrite(filename,B9,'Perceptron',f); end;


if ( z==1 && w==1) xlswrite(filename,C,'Regression LINEAL','A1'); end;
f=(z*3+w)-2; f=int2str(f); f=strcat('A',f);
if (  z>1 || w>1) xlswrite(filename,C,'Regression LINEAL',f); end;
    
%xlswrite(filename,C2,'Regression LINEAL','A4'); 
if ( z==1 && w==1) xlswrite(filename,D,'Regression NO LINEAL','A1'); end;
f=(z*3+w)-2; f=int2str(f); f=strcat('A',f);
if ( z>1 || w>1) xlswrite(filename,D,'Regression NO LINEAL',f); end;

 %xlswrite(filename,D2,'Regression NO LINEAL','A4'); 
%end;


    end;
 
end;
