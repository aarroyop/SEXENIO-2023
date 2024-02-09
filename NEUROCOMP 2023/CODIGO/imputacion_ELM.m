clear all; close all;
datos=52685;
entrada='todosELM.txt'; 
tabla=readtable(entrada);
dataelement=table2array(tabla);
for (j=1:datos)
   if isnan(dataelement(j,2)) dataelement(j,1)=1; end;
end;
%dataelement=norm(dataelement);
for (j=1:52000)
    clearvars -except h j dataelement k
    h=0;
    k=j;
    while (dataelement(k,1)==1)  %(:,1:8)
        dataelement(k,1)=0;
        %j=j+1;
        h=h+1;
        k=k+1;       
    end;
    if (h>0)
        %dataelemment2=dataelement(j,j+h,:);
        Y0=dataelement(1:j,2); %columna objetivo%
        X0=dataelement(1:j,3:8);
        XP=dataelement(j:j-1+h,3:8);  %la que funciona es XP=dataelement(j:j-1+h,3:8)
        cv = cvpartition(length(Y0),'Kfold',10)
        %X0=X0./norm(X0);
        hiddenLayerSizeP = 5;
        X1=X0(:,1); X2=X0(:,2); X3=X0(:,3); X4=X0(:,4); X5=X0(:,5); X6=X0(:,6);     
        cv = cvpartition(length(Y0),'Kfold',10)
        for i = 1:1
           trainIdxs{i} = find(training(cv,i)); %90 para entrenamiento
           testIdxs{i} =  find(test(cv,i));   %10 para test
            trainMatrix{i} = [X1(trainIdxs{i}) X2(trainIdxs{i}) X3(trainIdxs{i}) X4(trainIdxs{i}) X5(trainIdxs{i}) X6(trainIdxs{i}) Y0(trainIdxs{i})];
            testMatrix{i} = [X1(testIdxs{i}) X2(testIdxs{i}) X3(testIdxs{i}) X4(testIdxs{i}) X5(testIdxs{i}) X6(testIdxs{i}) Y0(testIdxs{i})];    
            trainFcnP = 'trainlm';  %     
            netP = fitnet(hiddenLayerSizeP,trainFcnP);
            netP.divideFcn='divideind';
          netP.divideParam.testInd = testIdxs{i};
           netP.divideParam.trainInd = trainIdxs{i};
           netP.trainParam.epochs = 10; 
           [netP, trP] = train(netP,trainMatrix{i}(:,1:6)',trainMatrix{i}(:,7)');

%          [yP] = netP(trainMatrix{j}(:,1:8)');
%                    
             %[TrainingTime, TestingTime, TrainingAccuracy(i), TestingAccuracy] = ELM(trainMatrix{i}(:,1:7),testMatrix{i}(:,1:7), 0, 8,'sig')
%            Opts.ELM_Type='Regrs';    % 'Class' for classification and 'Regrs' for regression
%             Opts.number_neurons=200;  % Maximam number of neurons
%             Opts.Tr_ratio=0.70;       % training ratio
%             Opts.Bn=1;   
%             net = elm_LB(X0,Y0, Opts);
        end;
        v=1;
        for(j=j:j-1+h)
         y3RBF=sim(netP,XP(v,:)'); %funciona para MLP 
         % y3RBF=predict(netP,XP');
        %y3RBF=elmPredict(net,XP); 
        
         dataelement(j,2)=y3RBF; %sim(netR,XP(h,:)');
         v=v+1;
        end;
    end;
    j=1;
end;
dataelement_salida=dataelement(:,2:8);
% dataelement_salida(:,[1 3])=dataelement(:,[2 4]);
% dataelement_salida(:,[3 1])=dataelement(:,[4 2]);
writematrix(dataelement_salida,'imputados_MLP.txt','Delimiter','tab');
% Y(j,1) ==0 end; 
%      for d=1:7
%         dataelement = fscanf(fd,'%f',1);
%         X(j,d) = dataelement;
%     end;
% else if Y(j,1) ==1 end;
%         dataelement=X;
% %         data21=dataelement;
% %         dataelement(:,[3 1])=data21(:,[1 3]);
% %         dataelement(:,[1 3])=data21(:,[3 1]);
%         dataelement=dataelement./norm(dataelement);
%         Y0=dataelement(:,1); %columna objetivo
%         X0=dataelement(:,2:7);
%         neurons=10;
%         X1=X0(:,1); X2=X0(:,2); X3=X0(:,3); X4=X0(:,4); X5=X0(:,5); X6=X0(:,6);     
%         cv = cvpartition(length(Y0),'Kfold',10)
%         lx=10;
%         for i = 1:lx %diez veces para los 10 kfolds
%      %común a perceptron, RB, regresión lineal y no lineal, elm y las de la
%      %app
%              hiddenLayerSizeP = neurons;
%              trainIdxs{i} = find(training(cv,i)); %90 para entrenamiento
%              testIdxs{i} =  find(test(cv,i));   %10 para test
%              trainMatrix{i} = [X1(trainIdxs{i}) X2(trainIdxs{i}) X3(trainIdxs{i}) X4(trainIdxs{i}) X5(trainIdxs{i}) X6(trainIdxs{i}) Y0(trainIdxs{i})];
%              testMatrix{i} = [X1(testIdxs{i}) X2(testIdxs{i}) X3(testIdxs{i}) X4(testIdxs{i}) X5(testIdxs{i}) X6(testIdxs{i}) Y0(testIdxs{i})];    
%              [TrainingTime, TestingTime, TrainingAccuracy(i), TestingAccuracy] = ELM(trainMatrix{i}(:,1:7),testMatrix{i}(:,1:7), 0, neurons,'sig');
%         end;
%         k=0;
%         for (h=j:datos)
%            Y2(h,1) = fscanf(fd,'%f',1);
%              if Y2(h,1) ==1 k=k+1;%hay que imputar esta fila 
%                for d=1:7
%                  dataelement = fscanf(fd,'%f',1);
%                  X2(h,d) = dataelement;
%         end;
% end;
% end;
% end;
% dataelement=[Y X];
% dataelement2=dataelement./norm(dataelement);
% for (zz=1:2)
%        if(zz==1) idx='City'; distance = 'cityblock'; end;
%        if(zz==2) idx='Cosine'; distance = 'cosine'; end; 
%        idx = kmeans(dataelement2, 3, 'Distance', distance);
%        %for (zzz=1:3)
%        k1i=1;
%        k2i=1;
%        k3i=1;
%        clear vars c1 c2 c3
%        for (k=1:datos)
%           if(idx(k)==1)
%             c1(k1i,:)=dataelement(k,:); k1i=k1i+1; end;
%           if(idx(k)==2)
%             c2(k2i,:)=dataelement(k,:); k2i=k2i+1; end;
%           if(idx(k)==3)
%             c3(k3i,:)=dataelement(k,:); k3i=k3i+1; end;
%       end;
%         for (h=1:3)
%            if(zz==1) idx2='city'; end;
%            if(zz==2) idx2='cosine'; end; 
%           mes =int2str(z); 
%           h2 =int2str(h);
%           
% %            if (z==1) season='inv'; end;
% %            if (z==2) season='pri'; end;
% %            if (z==3) season='ver'; end;
% %            if (z==4) season='oto'; end;
%            
%            if (h==1) cluster = c1; end;
%            if (h==2) cluster = c2; end;
%            if (h==3) cluster = c3; end;
%            %cluster=int2str(h);
%            salida=strcat('Mes',mes,'cluster',h2,'medida',idx2,'_lineal.txt');  
%             
%            writematrix(cluster,salida,'Delimiter','tab');
%            tamano(z, zz, h)=size(cluster,1);
%         end;
%    end;
% end;


