%function radiacion;
%fd = fopen('horasordenadossoco.txt','r');
%fd = fopen('aguirredias.txt','r');
close all;
fd = fopen('carmendias.txt','r');
%fd = fopen('todosdias.txt','r');
if (fd<0)
    'fopen failed'
    return;
end;

for (j=1:1012)
%for (j=1:253)
    toma(j) = fscanf(fd,'%d',1);
      
      
        for d=1:4
         dataelement = fscanf(fd,'%f',1);
         data(j,d) = dataelement;
        end;
   
end;

L=toma'
sD=data./norm(data)
%sD=pcaproj(sD,2)

%sD = som_read_data('carmen2.txt');
 sD = som_data_struct(sD,'name','Carmen','comp_names',{'NO2','SO2','CO','O3'});
% sD = som_label(sD,'add',1:184,'1');
% sD = som_label(sD,'add',185:368,'2');
% sD = som_label(sD,'add',369:552,'3');
% sD = som_label(sD,'add',553:736,'4');
% sD = som_label(sD,'add',737:920,'5');
% sD = som_label(sD,'add',921:1104,'6');
% sD = som_label(sD,'add',1105:1288,'7');
% sD = som_label(sD,'add',1289:1472,'8');
% sD = som_label(sD,'add',1473:1656,'9');
% sD = som_label(sD,'add',1657:1840,'10');
% sD = som_label(sD,'add',1841:2024,'11');

%  sD = som_label(sD,'add',1:92,'1');
% sD = som_label(sD,'add',93:184,'2');
% sD = som_label(sD,'add',185:277,'3');
% sD = som_label(sD,'add',278:369,'4');
% sD = som_label(sD,'add',370:463,'5');
% sD = som_label(sD,'add',464:556,'6');
% sD = som_label(sD,'add',557:649,'7');
% sD = som_label(sD,'add',650:742,'8');
% sD = som_label(sD,'add',743:835,'9');
% sD = som_label(sD,'add',836:928,'10');
% sD = som_label(sD,'add',929:1012,'11');


% %sD = som_normalize(sD,'var');
%sM = som_make(sD,'init','randinit','algorithm','batch','neign', 'gaussian', 'munits', 100)
% 
 U = som_umat(sM);
 Um = U(1:2:size(U,1),1:2:size(U,2));

%    A related technique is to assign colors to the map units such
%    that similar map units get similar colors.

%    Here, four clustering figures are shown: 
%     - U-matrix
%     - median distance matrix (with grayscale)
%     - median distance matrix (with map unit size)
%     - similarity coloring, made by spreading a colormap
%       on top of the principal component projection of the
%       prototype vectors

%  subplot(2,2,1)
%  h=som_cplane([sM.topol.lattice,'U'],sM.topol.msize, U(:)); 
%  set(h,'Edgecolor','none'); title('U-matrix')

figure(6)
 h=som_cplane(sM, Um(:));
% bmus = som_bmus(sM,sD,'best')
% som_trajectory(bmus)
%set(h,'Edgecolor','none'); title('D-matrix (grayscale)')

% subplot(2,2,3)
% som_show(sM,'comp',1-Um(:)/max(Um(:)))
%  title('D-matrix (marker size)')

% subplot(2,2,4)
% C = som_colorcode(Pm);  % Pm is the PC-projection calculated earlier
% som_cplane(sM,C)
% title('Similarity coloring')





%sM = som_make(sD,'init','randinit','algorithm','batch','neign', 'gaussian', 'msize',[25 8])%'cutgauss',%                          'ep' or 'bubble'
%sM = som_autolabel(sM,sD,'vote');
%figure(1)
%som_show(sM,'compi', 'all')
% som_grid(sM.codebook(:))
figure(8)
som_show(sM, 'empty', 'Hits','bar','none')
som_show_add('hit',som_hits(sM,sD),'EdgeColor', 'r','text','on', 'textcolor','k')
%  figure(2)
%  som_show(sM,'umat', 'all')
 %som_show(sM, 'comp')
%hold on
% figure(3)
% som_show(sM,'umat', 'all')
% som_show_add('hit',som_hits(sM,sD),'Marker','lattice','EdgeColor', 'none','text','on')

figure(4)
som_show(sM,'umat', 'all')
bmus = som_bmus(sM,sD,'best')
som_trajectory(bmus)

figure(5)
 som_show(sM,'comp', 'all','bar','none')
 som_trajectory(bmus,'data1', sD.data(:,[1 2 3 4]), 'name1', {'NO2'; 'SO2'; 'CO'; 'O3'} )
% 
% 
% close all
%hold off
%som_show(sM, 'comp','all')
%som_show(sM, 'comp', [1:4])
% f1=figure;
% [Pd,V,me,l] = pcaproj(sD,2); Pm = pcaproj(sM,V,me); % PC-projection
% Code = som_colorcode(Pm); % color coding
% hits = som_hits(sM,sD);  % hits
% U = som_umat(sM); % U-matrix
% Dm = U(1:2:size(U,1),1:2:size(U,2)); % distance matrix
% Dm = 1-Dm(:)/max(Dm(:)); Dm(find(hits==0)) = 0; % clustering info
% 
% subplot(1,3,1)
% som_cplane(sM,Code,Dm);
% hold on
% som_grid(sM,'Label',cellstr(int2str(hits)),...
% 	 'Line','none','Marker','none','Labelcolor','k');
% hold off 
% title('Color code')
% 
% subplot(1,3,2)
% som_grid(sM,'Coord',Pm,'MarkerColor',Code,'Linecolor','k');
% hold on, plot(Pd(:,1),Pd(:,2),'k+'), hold off, axis tight, axis equal
% title('PC projection')


%sD.data
%bmus = som_bmus(sM,sD,'best')
%som_trajectory(bmus)
%som_trajectory(bmus,'data1', sD.data(:,[1 2 3 4]), 'name1', {'NO2'; 'SO2'; 'CO'; 'O3'} )
%som_trajectory(bmus,'data1', sD.data(:,1), 'name1', {'NO2'} )
%som_trajectory(bmus,'data1', sD.data(:,[1 2 3]), 'name1', {'NO2'; 'SO2'; 'CO'} )





