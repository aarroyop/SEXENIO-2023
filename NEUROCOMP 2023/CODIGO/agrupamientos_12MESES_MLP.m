clear all; close all;
for z=1:12

  if (z==1) datos=4464; entrada='enero_MLP.txt'; end;
  if (z==2) datos=4032; entrada='febrero_MLP.txt'; end;
   if (z==3) datos=4464; entrada='marzo_MLP.txt';  end;
  if (z==4) datos=4284; entrada='abril_MLP.txt';  end;
  if (z==5) datos=4465; entrada='mayo_MLP.txt'; end;
  if (z==6) datos=4320; entrada='junio_MLP.txt';  end;
  if (z==7) datos=4464; entrada='julio_MLP.txt'; end;
  if (z==8) datos=4463; entrada='agosto_MLP.txt';  end;
  if (z==9) datos=4450; entrada='septiembre_MLP.txt'; end;
  if (z==10) datos=4458; entrada='octubre_MLP.txt';  end;
  if (z==11) datos=4320; entrada='noviembre_MLP.txt'; end;
  if (z==12) datos=4464; entrada='diciembre_MLP.txt'; end;
fd = fopen(entrada,'r');
%filename='resultados_LINEAR.xlsx';
mes=z;
if (fd<0)
    'fopen failed'
    return;
end;
for (j=1:datos)
     Y(j,1) = fscanf(fd,'%f',1);
    for d=1:6
        dataelement = fscanf(fd,'%f',1);
        X(j,d) = dataelement;
    end;
end;
dataelement=[Y X];
dataelement2=dataelement./norm(dataelement);
for (zz=1:2)
       if(zz==1) idx='City'; distance = 'cityblock'; end;
       if(zz==2) idx='Cosine'; distance = 'cosine'; end; 
       idx = kmeans(dataelement2, 3, 'Distance', distance);
       %for (zzz=1:3)
       k1i=1;
       k2i=1;
       k3i=1;
       clear vars c1 c2 c3
       for (k=1:datos)
          if(idx(k)==1)
            c1(k1i,:)=dataelement(k,:); k1i=k1i+1; end;
          if(idx(k)==2)
            c2(k2i,:)=dataelement(k,:); k2i=k2i+1; end;
          if(idx(k)==3)
            c3(k3i,:)=dataelement(k,:); k3i=k3i+1; end;
      end;
        for (h=1:3)
           if(zz==1) idx2='city'; end;
           if(zz==2) idx2='cosine'; end; 
          mes =int2str(z); 
          h2 =int2str(h);
          
%            if (z==1) season='inv'; end;
%            if (z==2) season='pri'; end;
%            if (z==3) season='ver'; end;
%            if (z==4) season='oto'; end;
           
           if (h==1) cluster = c1; end;
           if (h==2) cluster = c2; end;
           if (h==3) cluster = c3; end;
           %cluster=int2str(h);
           salida=strcat('Mes',mes,'cluster',h2,'medida',idx2,'_MLP.txt');  
            
           writematrix(cluster,salida,'Delimiter','tab');
           tamano(z, zz, h)=size(cluster,1);
        end;
   end;
end;

