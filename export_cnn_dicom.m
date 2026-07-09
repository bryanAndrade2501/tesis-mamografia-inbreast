%% export_cnn_dicom.m
% Exporta las 410 imágenes del pipeline como DICOM en results_images/:
%   entrada/          -> X  (preprocesamiento + denoise)
%   referencia_clahe/ -> Y  (preprocesamiento + denoise + CLAHE)
%   cnn/              -> Y^ (salida de la red entrenada)
%
% Requisitos:
%   - red_enhancement_inbreast.mat y dataset_cache.mat
%   - INbreast AllDICOMs (metadatos DICOM originales)
%
% Uso:
%   export_cnn_dicom

clear; clc;

%% Configuración
cfg = struct();
cfg.tesisRoot      = 'C:\Users\Asus\Documents\Tesis';
cfg.inbreastRoot   = fullfile(cfg.tesisRoot, 'imagenes', 'INbreast Release 1.0');
cfg.dicomFolder    = fullfile(cfg.inbreastRoot, 'AllDICOMs');
cfg.modelPath      = fullfile(cfg.tesisRoot, 'RESULTADOS', 'EX2', 'red_enhancement_inbreast.mat');
cfg.cachePath      = fullfile(cfg.tesisRoot, 'RESULTADOS', 'EX2', 'dataset_cache.mat');
cfg.outputRoot     = fullfile(cfg.tesisRoot, 'results_images');
cfg.folderEntrada  = fullfile(cfg.outputRoot, 'entrada');
cfg.folderClahe    = fullfile(cfg.outputRoot, 'referencia_clahe');
cfg.folderCnn      = fullfile(cfg.outputRoot, 'cnn');
cfg.executionEnvironment = "cpu";
cfg.dicomBitsStored = 16;
cfg.miniBatchSize  = 4;
cfg.exportCnn      = true;
cfg.exportEntrada  = true;
cfg.exportClahe    = true;

assert(isfile(cfg.modelPath), 'No se encontró el modelo: %s', cfg.modelPath);
assert(isfile(cfg.cachePath), 'No se encontró dataset_cache: %s', cfg.cachePath);
assert(isfolder(cfg.dicomFolder), 'No se encontró AllDICOMs: %s', cfg.dicomFolder);

for folder = {cfg.outputRoot, cfg.folderEntrada, cfg.folderClahe, cfg.folderCnn}
    if ~exist(folder{1}, 'dir')
        mkdir(folder{1});
    end
end

fprintf('=== Exportación DICOM (entrada / CLAHE / CNN) ===\n');
fprintf('Modelo: %s\n', cfg.modelPath);
fprintf('Salida: %s\n', cfg.outputRoot);

%% Cargar modelo y dataset cacheado
loadedNet = load(cfg.modelPath, 'net');
net = loadedNet.net;

cache = load(cfg.cachePath, 'X', 'Y', 'fileNames', 'patientIds');
X = cache.X;
Y = cache.Y;
fileNames = cache.fileNames;
patientIds = cache.patientIds;
numImages = size(X, 4);

fprintf('Imágenes en cache: %d\n', numImages);

%% Índice de DICOM originales
dicomFiles = dir(fullfile(cfg.dicomFolder, '**', '*.dcm'));
nameToPath = containers.Map('KeyType', 'char', 'ValueType', 'char');
for k = 1:numel(dicomFiles)
    nameToPath(dicomFiles(k).name) = fullfile(dicomFiles(k).folder, dicomFiles(k).name);
end

%% Exportación imagen por imagen
exportLog = table('Size', [numImages 6], ...
    'VariableTypes', {'double', 'string', 'string', 'string', 'string', 'string'}, ...
    'VariableNames', { ...
        'Idx', 'ArchivoOriginal', 'ArchivoEntrada', 'ArchivoCLAHE', 'ArchivoCNN', 'Paciente'});

missingSources = strings(0, 1);

fprintf('Procesando y exportando DICOM...\n');

for i = 1:numImages
    if mod(i, 25) == 0 || i == 1 || i == numImages
        fprintf('  %d / %d\n', i, numImages);
    end

    srcName = char(fileNames{i});
    if ~isKey(nameToPath, srcName)
        missingSources(end + 1, 1) = string(srcName); %#ok<AGROW>
        warning('DICOM original no encontrado: %s', srcName);
        continue;
    end

    srcPath = nameToPath(srcName);
    baseTag = erase(srcName, '.dcm');

    nameEntrada = [baseTag, '_INPUT.dcm'];
    nameClahe   = [baseTag, '_CLAHE.dcm'];
    nameCnn     = [baseTag, '_CNN.dcm'];

    if cfg.exportEntrada
        writeProcessedDicom(squeeze(X(:, :, 1, i)), srcPath, ...
            fullfile(cfg.folderEntrada, nameEntrada), cfg, ...
            'Entrada pipeline (preproc+denoise)', ...
            'Imagen de entrada X: preprocesamiento y filtro mediana 3x3.');
    end

    if cfg.exportClahe
        writeProcessedDicom(squeeze(Y(:, :, 1, i)), srcPath, ...
            fullfile(cfg.folderClahe, nameClahe), cfg, ...
            'Referencia CLAHE', ...
            'Objetivo supervisado Y: preprocesamiento, denoise y CLAHE.');
    end

    if cfg.exportCnn
        Xi = X(:, :, :, i);
        Yhat = predict(net, Xi, ...
            'ExecutionEnvironment', cfg.executionEnvironment, ...
            'MiniBatchSize', cfg.miniBatchSize);
        writeProcessedDicom(Yhat, srcPath, ...
            fullfile(cfg.folderCnn, nameCnn), cfg, ...
            'CNN Enhancement - U-Net INbreast', ...
            'Salida CNN: mejora aprendida sobre imagen filtrada.');
    end

    exportLog.Idx(i) = i;
    exportLog.ArchivoOriginal(i) = string(srcName);
    exportLog.ArchivoEntrada(i) = string(nameEntrada);
    exportLog.ArchivoCLAHE(i) = string(nameClahe);
    exportLog.ArchivoCNN(i) = string(nameCnn);
    exportLog.Paciente(i) = string(patientIds{i});
end

writetable(exportLog, fullfile(cfg.outputRoot, 'export_log.csv'));

nEntrada = numel(dir(fullfile(cfg.folderEntrada, '*_INPUT.dcm')));
nClahe   = numel(dir(fullfile(cfg.folderClahe, '*_CLAHE.dcm')));
nCnn     = numel(dir(fullfile(cfg.folderCnn, '*_CNN.dcm')));

fprintf('\nExportación finalizada.\n');
fprintf('  entrada/          : %d DICOM\n', nEntrada);
fprintf('  referencia_clahe/ : %d DICOM\n', nClahe);
fprintf('  cnn/              : %d DICOM\n', nCnn);
fprintf('  Log: %s\n', fullfile(cfg.outputRoot, 'export_log.csv'));

if ~isempty(missingSources)
    fprintf('  Advertencia: %d archivos originales no encontrados.\n', numel(missingSources));
    writetable(table(missingSources, 'VariableNames', {'ArchivoFaltante'}), ...
        fullfile(cfg.outputRoot, 'archivos_faltantes.csv'));
end

%% ========================================================================
%  FUNCIONES LOCALES
% ========================================================================

function writeProcessedDicom(Inorm, srcDicomPath, outDicomPath, cfg, seriesDescription, imageComments)
    Inorm = single(Inorm);
    Inorm = min(max(Inorm, 0), 1);

    info = dicominfo(srcDicomPath);
    pixelData = normalizedToUintPixel(Inorm, cfg.dicomBitsStored);

    info.Rows = size(pixelData, 1);
    info.Columns = size(pixelData, 2);
    info.SamplesPerPixel = 1;
    info.PhotometricInterpretation = 'MONOCHROME2';
    info.BitsAllocated = cfg.dicomBitsStored;
    info.BitsStored = cfg.dicomBitsStored;
    info.HighBit = cfg.dicomBitsStored - 1;
    info.PixelRepresentation = 0;
    info.RescaleSlope = 1;
    info.RescaleIntercept = 0;
    info.WindowCenter = double(round(mean(pixelData(:), 'omitnan')));
    info.WindowWidth = double(max(pixelData(:)) - min(pixelData(:)) + 1);

    if isfield(info, 'PixelData')
        info = rmfield(info, 'PixelData');
    end

    info.SeriesDescription = seriesDescription;
    info.ImageComments = imageComments;
    info.SeriesInstanceUID = dicomuid;
    info.SOPInstanceUID = dicomuid;

    dicomwrite(pixelData, outDicomPath, info, 'CreateMode', 'copy');
end

function pixelData = normalizedToUintPixel(Inorm, bitsStored)
    switch bitsStored
        case 8
            pixelData = im2uint8(Inorm);
        case 16
            pixelData = uint16(round(double(Inorm) * 65535));
        otherwise
            error('bitsStored no soportado: %d', bitsStored);
    end
end
