dicomFile = 'C:\Users\Asus\Documents\Tesis\imagenes\INbreast Release 1.0\AllDICOMs\20588680_036aff49b8ac84f0_MG_L_ML_ANON.dcm';
jpgFile   = 'salida.jpg';

info = dicominfo(dicomFile);
I    = dicomread(info);

% Escala a 8 bits para JPG
Ijpg = im2uint8(mat2gray(I));

imwrite(Ijpg, jpgFile, 'Quality', 95);
disp(['Guardado: ', jpgFile]);