%% TESIS: Mejora de imágenes mamográficas DICOM (INbreast) mediante CNN
% Secciones cubiertas:
%   3.4.1 Adquisición de imágenes
%   3.4.2 Preprocesamiento
%   3.4.3 Reducción de ruido
%   3.4.4 Mejora de contraste
%   3.4.5 Modelo de red neuronal
%   3.4.6 Evaluación de resultados
%
% Dataset: INbreast Release 1.0 (Kaggle / repositorio oficial)
% Herramienta: MATLAB + Deep Learning Toolbox + Image Processing Toolbox

clear; clc; close all;

%% ========================================================================
%  CONFIGURACIÓN GENERAL
% ========================================================================
cfg = struct();

% --- Rutas (ajustar según tu equipo) ---
cfg.inbreastRoot   = 'C:\Users\naria\Documents\bryan\modelo\data';
cfg.dicomFolder    = fullfile(cfg.inbreastRoot, 'all-mias');
cfg.csvMetadata    = fullfile(cfg.inbreastRoot, 'INbreast.csv');
cfg.resultsFolder  = fullfile(pwd, 'resultados_inbreast_enhancement');

% --- Imagen y partición ---
cfg.imageSize      = [512 512];
cfg.trainRatio     = 0.70;
cfg.valRatio       = 0.15;
cfg.testRatio      = 0.15;
cfg.randomSeed     = 42;
cfg.splitByPatient = true;          % Evita fuga: vistas del mismo paciente en un solo split

% --- 3.4.2 Preprocesamiento ---
cfg.cropBreastROI      = true;
cfg.roiMarginPx        = 8;
cfg.normLowPercentile  = 0.5;
cfg.normHighPercentile = 99.5;

% --- 3.4.3 Reducción de ruido ---
cfg.denoiseMethod      = 'median';  % 'median' | 'bilateral' | 'none'
cfg.medianKernel       = [3 3];
cfg.bilateralDegree    = 3;
cfg.bilateralSpatial   = 3;

% --- 3.4.4 Mejora de contraste (pseudo-target supervisado) ---
cfg.useCLAHEasTarget   = true;
cfg.claheNumTiles      = [8 8];
cfg.claheClipLimit     = 0.01;
cfg.claheNBins         = 256;

% --- 3.4.5 Entrenamiento CNN ---
cfg.maxEpochs              = 80;
cfg.miniBatchSize          = 8;
cfg.initialLearnRate       = 1e-3;
cfg.learnRateDropFactor    = 0.5;
cfg.learnRateDropPeriod    = 20;
cfg.validationFrequency    = 25;
cfg.validationPatience     = 10;
cfg.executionEnvironment   = "auto";
cfg.doDataAugmentation     = true;
cfg.l2Regularization       = 1e-4;

% --- 3.4.6 Evaluación ---
cfg.savePreviewCount       = 10;
cfg.otsuScaleForeground    = 1.00;
cfg.minMaskArea            = 200;
cfg.compareClaheBaseline   = true;  % Línea base: denoise + CLAHE sin CNN

rng(cfg.randomSeed);

if ~exist(cfg.resultsFolder, 'dir')
    mkdir(cfg.resultsFolder);
end

diary(fullfile(cfg.resultsFolder, 'log_entrenamiento.txt'));
fprintf('=== Pipeline INbreast enhancement | %s ===\n', datestr(now));

%% ========================================================================
%  3.4.1 ADQUISICIÓN DE IMÁGENES
% ========================================================================
fprintf('\n--- 3.4.1 Adquisición de imágenes ---\n');

dicomFiles = dir(fullfile(cfg.dicomFolder, '**', '*.dcm'));
assert(~isempty(dicomFiles), ...
    'No se encontraron archivos .dcm en: %s', cfg.dicomFolder);

filePaths = arrayfun(@(f) fullfile(f.folder, f.name), dicomFiles, 'UniformOutput', false);
fileNames = {dicomFiles.name}';
numImages = numel(filePaths);

% Metadatos INbreast (opcional)
metadataTable = loadInbreastMetadata(cfg.csvMetadata);

% ID de paciente desde nombre DICOM (ej. 20586908_..._MG_R_CC_ANON.dcm)
patientIds = cell(numImages, 1);
for i = 1:numImages
    patientIds{i} = extractPatientId(fileNames{i});
end

acquisitionReport = table( ...
    (1:numImages)', string(fileNames), string(patientIds), ...
    'VariableNames', {'Idx', 'Archivo', 'Paciente'});

writetable(acquisitionReport, fullfile(cfg.resultsFolder, 'adquisicion_archivos.csv'));
fprintf('Imágenes DICOM encontradas: %d\n', numImages);
fprintf('Pacientes únicos: %d\n', numel(unique(patientIds)));

%% ========================================================================
%  3.4.2 – 3.4.4 CONSTRUCCIÓN DEL DATASET (preproc + denoise + contraste)
% ========================================================================
fprintf('\n--- 3.4.2 Preprocesamiento | 3.4.3 Denoise | 3.4.4 Contraste ---\n');

X = zeros([cfg.imageSize 1 numImages], 'single');   % Entrada CNN (preproc + denoise)
Y = zeros([cfg.imageSize 1 numImages], 'single');   % Target (preproc + denoise + CLAHE)
Xraw = zeros([cfg.imageSize 1 numImages], 'single'); % Solo preprocesado (sin denoise)
YclaheOnly = zeros([cfg.imageSize 1 numImages], 'single'); % Baseline CLAHE

preprocLog = cell(numImages, 1);

for i = 1:numImages
    if mod(i, 25) == 0 || i == 1 || i == numImages
        fprintf('  Procesando %d / %d\n', i, numImages);
    end

    [Iraw, meta] = readDicomMammography(filePaths{i});
    Iprep = preprocessMammography(Iraw, cfg, meta);
    Xraw(:,:,1,i) = im2single(Iprep);

    Iden = applyDenoising(Iprep, cfg);
    X(:,:,1,i) = im2single(Iden);

    if cfg.useCLAHEasTarget
        T = applyCLAHE(Iden, cfg);
    else
        error('Para esta tesis se requiere pseudo-target CLAHE (useCLAHEasTarget=true).');
    end
    Y(:,:,1,i) = im2single(T);
    YclaheOnly(:,:,1,i) = Y(:,:,1,i);

    preprocLog{i} = meta;
end

save(fullfile(cfg.resultsFolder, 'dataset_cache.mat'), ...
    'X', 'Y', 'Xraw', 'YclaheOnly', 'fileNames', 'patientIds', 'cfg', '-v7.3');
fprintf('Dataset construido y cacheado.\n');

%% ========================================================================
%  PARTICIÓN TRAIN / VAL / TEST (por paciente si está activado)
% ========================================================================
if cfg.splitByPatient
    [idxTrain, idxVal, idxTest] = splitIndicesByPatient( ...
        patientIds, cfg.trainRatio, cfg.valRatio, cfg.randomSeed);
else
    idx = randperm(numImages);
    nTrain = floor(cfg.trainRatio * numImages);
    nVal   = floor(cfg.valRatio   * numImages);
    idxTrain = idx(1:nTrain);
    idxVal   = idx(nTrain+1 : nTrain+nVal);
    idxTest  = idx(nTrain+nVal+1 : end);
end

nTrain = numel(idxTrain);
nVal   = numel(idxVal);
nTest  = numel(idxTest);

XTrain = X(:,:,:,idxTrain);   YTrain = Y(:,:,:,idxTrain);
XVal   = X(:,:,:,idxVal);     YVal   = Y(:,:,:,idxVal);
XTest  = X(:,:,:,idxTest);    YTest  = Y(:,:,:,idxTest);

testNames   = fileNames(idxTest);
testPatients = patientIds(idxTest);

fprintf('Partición | Train: %d | Val: %d | Test: %d\n', nTrain, nVal, nTest);

%% ========================================================================
%  AUMENTO DE DATOS (solo entrenamiento)
% ========================================================================
if cfg.doDataAugmentation
    [XTrain, YTrain] = augmentTrainingPairs(XTrain, YTrain);
    fprintf('Train tras aumento: %d muestras\n', size(XTrain, 4));
end

%% ========================================================================
%  3.4.5 MODELO DE RED NEURONAL (encoder-decoder + BatchNorm)
% ========================================================================
fprintf('\n--- 3.4.5 Modelo CNN ---\n');

layers = buildEnhancementUNet(cfg.imageSize);
analyzeNetworkIfPossible(layers);

checkpointDir = fullfile(cfg.resultsFolder, 'checkpoints');
if ~exist(checkpointDir, 'dir')
    mkdir(checkpointDir);
end

opts = trainingOptions('adam', ...
    'MaxEpochs', cfg.maxEpochs, ...
    'MiniBatchSize', cfg.miniBatchSize, ...
    'InitialLearnRate', cfg.initialLearnRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', cfg.learnRateDropFactor, ...
    'LearnRateDropPeriod', cfg.learnRateDropPeriod, ...
    'L2Regularization', cfg.l2Regularization, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', cfg.validationFrequency, ...
    'ValidationPatience', cfg.validationPatience, ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ExecutionEnvironment', cfg.executionEnvironment, ...
    'CheckpointPath', checkpointDir);

fprintf('Entrenando red...\n');
net = trainNetwork(XTrain, YTrain, layers, opts);

save(fullfile(cfg.resultsFolder, 'red_enhancement_inbreast.mat'), 'net', 'cfg', '-v7.3');

%% ========================================================================
%  3.4.6 EVALUACIÓN DE RESULTADOS
% ========================================================================
fprintf('\n--- 3.4.6 Evaluación ---\n');

YPred = predict(net, XTest, 'ExecutionEnvironment', cfg.executionEnvironment);

results = evaluateEnhancementResults(XTest, YTest, YPred, testNames, testPatients, cfg);

writetable(results.perImage, fullfile(cfg.resultsFolder, 'metricas_test.csv'));
writetable(results.summary, fullfile(cfg.resultsFolder, 'resumen_metricas.csv'));

validationReport = struct();
validationReport.numTrain = size(XTrain, 4);
validationReport.numVal   = size(XVal, 4);
validationReport.numTest  = nTest;
validationReport.summary    = table2struct(results.summary);
validationReport.passPSNR   = results.summary.Mean_Delta_PSNR > 0;
validationReport.passSSIM   = results.summary.Mean_Delta_SSIM > 0;
validationReport.passCNR    = results.summary.Mean_Delta_CNR  > 0;
validationReport.conclusion = "Mejora confirmada si las deltas CNN vs entrada son positivas.";

save(fullfile(cfg.resultsFolder, 'validation_report.mat'), 'validationReport');
savejsonIfPossible(fullfile(cfg.resultsFolder, 'validation_report.json'), validationReport);

disp(results.summary);

saveQualitativePreviews(XTest, YTest, YPred, testNames, cfg);
makeMetricPlots(results.perImage, cfg.resultsFolder);

fprintf('\n=== Pipeline finalizado | Resultados en: %s ===\n', cfg.resultsFolder);
diary off;

%% ========================================================================
%  FUNCIONES LOCALES
% ========================================================================

function metaTable = loadInbreastMetadata(csvPath)
    metaTable = table();
    if ~isfile(csvPath)
        warning('No se encontró INbreast.csv en: %s', csvPath);
        return;
    end
    try
        opts = detectImportOptions(csvPath, 'Delimiter', ';');
        metaTable = readtable(csvPath, opts);
    catch ME
        warning('No se pudo leer metadatos: %s', ME.message);
    end
end

function patientId = extractPatientId(fileName)
    parts = split(fileName, '_');
    if numel(parts) >= 1
        patientId = char(parts(1));
    else
        [~, patientId, ~] = fileparts(fileName);
    end
end

function [I, meta] = readDicomMammography(filePath)
    % Lectura DICOM con correcciones habituales en mamografía.
    info = struct();
    try
        info = dicominfo(filePath);
    catch
        warning('Metadata DICOM limitada: %s', filePath);
    end

    I = double(dicomread(filePath));

    if isfield(info, 'RescaleSlope')
        slope = double(info.RescaleSlope);
    else
        slope = 1;
    end
    if isfield(info, 'RescaleIntercept')
        intercept = double(info.RescaleIntercept);
    else
        intercept = 0;
    end
    I = I * slope + intercept;

    if isfield(info, 'PhotometricInterpretation') && ...
            strcmpi(info.PhotometricInterpretation, 'MONOCHROME1')
        I = max(I(:)) - I;
    end

    meta = struct();
    meta.filePath = filePath;
    if isfield(info, 'Modality'), meta.modality = info.Modality; end
    if isfield(info, 'ViewPosition'), meta.view = info.ViewPosition; end
    if isfield(info, 'Rows'), meta.rows = info.Rows; end
    if isfield(info, 'Columns'), meta.columns = info.Columns; end
end

function Iout = preprocessMammography(I, cfg, ~)
    I = double(I);
    I(I < 0) = 0;

    if cfg.cropBreastROI
        I = cropBreastBoundingBox(I, cfg.roiMarginPx);
    end

    I = imresize(I, cfg.imageSize, 'bilinear');

    lo = prctile(I(:), cfg.normLowPercentile);
    hi = prctile(I(:), cfg.normHighPercentile);
    if hi <= lo
        Iout = mat2gray(I);
    else
        Iout = (I - lo) / (hi - lo);
        Iout = min(max(Iout, 0), 1);
    end
end

function Icrop = cropBreastBoundingBox(I, marginPx)
    mask = I > max(prctile(I(:), [1 99])) * 0.02;
    mask = bwareaopen(mask, 500);
    mask = imfill(mask, 'holes');

    if nnz(mask) < 100
        Icrop = I;
        return;
    end

    stats = regionprops(mask, 'BoundingBox');
    [~, idx] = max(cellfun(@(bb) bb(3)*bb(4), {stats.BoundingBox}));
    bb = stats(idx).BoundingBox;

    r1 = max(1, floor(bb(2)) - marginPx);
    c1 = max(1, floor(bb(1)) - marginPx);
    r2 = min(size(I,1), ceil(bb(2) + bb(4)) + marginPx);
    c2 = min(size(I,2), ceil(bb(1) + bb(3)) + marginPx);

    Icrop = I(r1:r2, c1:c2);
end

function Iout = applyDenoising(I, cfg)
    switch lower(cfg.denoiseMethod)
        case 'median'
            Iout = medfilt2(I, cfg.medianKernel);
        case 'bilateral'
            Iout = imbilatfilt(I, cfg.bilateralDegree, cfg.bilateralSpatial);
        case 'none'
            Iout = I;
        otherwise
            error('Método de denoise no soportado: %s', cfg.denoiseMethod);
    end
    Iout = min(max(Iout, 0), 1);
end

function Iout = applyCLAHE(I, cfg)
    Iout = adapthisteq(I, ...
        'NumTiles', cfg.claheNumTiles, ...
        'ClipLimit', cfg.claheClipLimit, ...
        'NBins', cfg.claheNBins);
    Iout = min(max(im2single(Iout), 0), 1);
end

function [idxTrain, idxVal, idxTest] = splitIndicesByPatient(patientIds, trainRatio, valRatio, seed)
    patients = unique(patientIds, 'stable');
    nPatients = numel(patients);

    rng(seed);
    perm = randperm(nPatients);

    nTrainP = max(1, floor(trainRatio * nPatients));
    nValP   = max(1, floor(valRatio * nPatients));
    if nTrainP + nValP >= nPatients
        nValP = max(1, nPatients - nTrainP - 1);
    end
    nTestP = nPatients - nTrainP - nValP;

    trainPatients = patients(perm(1:nTrainP));
    valPatients   = patients(perm(nTrainP+1 : nTrainP+nValP));
    testPatients  = patients(perm(nTrainP+nValP+1 : end));

    idxTrain = find(ismember(patientIds, trainPatients));
    idxVal   = find(ismember(patientIds, valPatients));
    idxTest  = find(ismember(patientIds, testPatients));

    fprintf('Split por paciente | Train: %d pac (%d img) | Val: %d pac (%d img) | Test: %d pac (%d img)\n', ...
        nTrainP, numel(idxTrain), nValP, numel(idxVal), nTestP, numel(idxTest));
end

function layers = buildEnhancementUNet(imageSize)
    % U-Net compacta para restitución espacial tras pooling.
    layers = [
        imageInputLayer([imageSize 1], 'Name', 'input', 'Normalization', 'none')

        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'enc1_conv1')
        batchNormalizationLayer('Name', 'enc1_bn1')
        reluLayer('Name', 'enc1_relu1')
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'enc1_conv2')
        batchNormalizationLayer('Name', 'enc1_bn2')
        reluLayer('Name', 'enc1_relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'enc2_conv1')
        batchNormalizationLayer('Name', 'enc2_bn1')
        reluLayer('Name', 'enc2_relu1')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'enc2_conv2')
        batchNormalizationLayer('Name', 'enc2_bn2')
        reluLayer('Name', 'enc2_relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'bottleneck_conv1')
        batchNormalizationLayer('Name', 'bottleneck_bn1')
        reluLayer('Name', 'bottleneck_relu1')
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'bottleneck_conv2')
        batchNormalizationLayer('Name', 'bottleneck_bn2')
        reluLayer('Name', 'bottleneck_relu2')

        transposedConv2dLayer(4, 64, 'Stride', 2, 'Cropping', 'same', 'Name', 'up1')
        batchNormalizationLayer('Name', 'up1_bn')
        reluLayer('Name', 'up1_relu')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'dec1_conv')
        batchNormalizationLayer('Name', 'dec1_bn')
        reluLayer('Name', 'dec1_relu')

        transposedConv2dLayer(4, 32, 'Stride', 2, 'Cropping', 'same', 'Name', 'up2')
        batchNormalizationLayer('Name', 'up2_bn')
        reluLayer('Name', 'up2_relu')
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'dec2_conv')
        batchNormalizationLayer('Name', 'dec2_bn')
        reluLayer('Name', 'dec2_relu')

        convolution2dLayer(3, 1, 'Padding', 'same', 'Name', 'conv_out')
        regressionLayer('Name', 'output')
    ];
end

function [XA, YA] = augmentTrainingPairs(X, Y)
    n = size(X, 4);
    XA = X;
    YA = Y;

    for i = 1:n
        xi = X(:,:,:,i);
        yi = Y(:,:,:,i);

        XA(:,:,:,end+1) = flip(xi, 2); %#ok<AGROW>
        YA(:,:,:,end+1) = flip(yi, 2); %#ok<AGROW>

        xi2 = circshift(xi, [0 2]);
        yi2 = circshift(yi, [0 2]);
        XA(:,:,:,end+1) = xi2; %#ok<AGROW>
        YA(:,:,:,end+1) = yi2; %#ok<AGROW>
    end
end

function results = evaluateEnhancementResults(XTest, YTest, YPred, testNames, testPatients, cfg)
    nTest = size(XTest, 4);

    perImage = table('Size', [nTest 13], ...
        'VariableTypes', [repmat({'string'}, 1, 2), repmat({'double'}, 1, 11)], ...
        'VariableNames', { ...
            'Imagen', 'Paciente', ...
            'PSNR_Input', 'PSNR_CNN', ...
            'SSIM_Input', 'SSIM_CNN', ...
            'MAE_Input', 'MAE_CNN', ...
            'CNR_Input', 'CNR_CNN', ...
            'Delta_PSNR', 'Delta_SSIM', 'Delta_CNR'});

    for i = 1:nTest
        Iin  = clip01(squeeze(XTest(:,:,1,i)));
        Itgt = clip01(squeeze(YTest(:,:,1,i)));
        Iprd = clip01(squeeze(YPred(:,:,1,i)));

        perImage.Imagen(i)   = string(testNames{i});
        perImage.Paciente(i) = string(testPatients{i});

        perImage.PSNR_Input(i) = psnr(Iin, Itgt);
        perImage.PSNR_CNN(i)   = psnr(Iprd, Itgt);
        perImage.SSIM_Input(i) = ssim(Iin, Itgt);
        perImage.SSIM_CNN(i)   = ssim(Iprd, Itgt);
        perImage.MAE_Input(i)  = mean(abs(Iin(:) - Itgt(:)), 'omitnan');
        perImage.MAE_CNN(i)    = mean(abs(Iprd(:) - Itgt(:)), 'omitnan');
        perImage.CNR_Input(i)  = computeAutomaticCNR(Iin, cfg);
        perImage.CNR_CNN(i)    = computeAutomaticCNR(Iprd, cfg);

        perImage.Delta_PSNR(i) = perImage.PSNR_CNN(i) - perImage.PSNR_Input(i);
        perImage.Delta_SSIM(i) = perImage.SSIM_CNN(i) - perImage.SSIM_Input(i);
        perImage.Delta_CNR(i)  = perImage.CNR_CNN(i)  - perImage.CNR_Input(i);
    end

    summary = table( ...
        mean(perImage.PSNR_Input, 'omitnan'), mean(perImage.PSNR_CNN, 'omitnan'), ...
        mean(perImage.SSIM_Input, 'omitnan'), mean(perImage.SSIM_CNN, 'omitnan'), ...
        mean(perImage.MAE_Input, 'omitnan'),  mean(perImage.MAE_CNN, 'omitnan'), ...
        mean(perImage.CNR_Input, 'omitnan'),  mean(perImage.CNR_CNN, 'omitnan'), ...
        mean(perImage.Delta_PSNR, 'omitnan'), mean(perImage.Delta_SSIM, 'omitnan'), ...
        mean(perImage.Delta_CNR, 'omitnan'), ...
        'VariableNames', { ...
            'Mean_PSNR_Input', 'Mean_PSNR_CNN', ...
            'Mean_SSIM_Input', 'Mean_SSIM_CNN', ...
            'Mean_MAE_Input', 'Mean_MAE_CNN', ...
            'Mean_CNR_Input', 'Mean_CNR_CNN', ...
            'Mean_Delta_PSNR', 'Mean_Delta_SSIM', 'Mean_Delta_CNR'});

    results = struct('perImage', perImage, 'summary', summary);
end

function Iout = clip01(I)
    Iout = min(max(I, 0), 1);
end

function cnrValue = computeAutomaticCNR(I, cfg)
    I = mat2gray(I);

    breastMask = I > graythresh(I) * 0.25;
    breastMask = bwareaopen(breastMask, cfg.minMaskArea);
    breastMask = imfill(breastMask, 'holes');

    if nnz(breastMask) < 100
        cnrValue = NaN;
        return;
    end

    breastPixels = I(breastMask);
    th = graythresh(breastPixels) * cfg.otsuScaleForeground;
    lesionMask = false(size(I));
    lesionMask(breastMask) = I(breastMask) > th;
    lesionMask = bwareaopen(lesionMask, cfg.minMaskArea);
    backgroundMask = breastMask & ~lesionMask;

    if nnz(lesionMask) < 50 || nnz(backgroundMask) < 50
        cnrValue = NaN;
        return;
    end

    mu1 = mean(I(lesionMask), 'omitnan');
    mu0 = mean(I(backgroundMask), 'omitnan');
    s1  = std(I(lesionMask), 0, 'omitnan');
    s0  = std(I(backgroundMask), 0, 'omitnan');
    denom = sqrt(s1^2 + s0^2);

    if denom < eps
        cnrValue = NaN;
    else
        cnrValue = abs(mu1 - mu0) / denom;
    end
end

function saveQualitativePreviews(XTest, YTest, YPred, testNames, cfg)
    previewFolder = fullfile(cfg.resultsFolder, 'previews');
    if ~exist(previewFolder, 'dir')
        mkdir(previewFolder);
    end

    nPreview = min(cfg.savePreviewCount, size(XTest, 4));

    for i = 1:nPreview
        Iin  = squeeze(XTest(:,:,1,i));
        Itgt = squeeze(YTest(:,:,1,i));
        Iprd = squeeze(YPred(:,:,1,i));

        f = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 1500 420]);

        subplot(1, 3, 1); imshow(Iin, []);  title('Entrada (preproc+denoise)');
        subplot(1, 3, 2); imshow(Iprd, []);  title('Salida CNN');
        subplot(1, 3, 3); imshow(Itgt, []);  title('Referencia CLAHE');

        sgtitle(sprintf('Test: %s', testNames{i}), 'Interpreter', 'none');
        exportgraphics(f, fullfile(previewFolder, sprintf('preview_%02d.png', i)));
        close(f);
    end
end

function makeMetricPlots(results, resultsFolder)
    f1 = figure('Visible', 'off', 'Color', 'w');
    boxplot([results.PSNR_Input, results.PSNR_CNN], 'Labels', {'Entrada', 'CNN'});
    ylabel('PSNR (dB)'); title('PSNR respecto a referencia CLAHE');
    exportgraphics(f1, fullfile(resultsFolder, 'boxplot_psnr.png')); close(f1);

    f2 = figure('Visible', 'off', 'Color', 'w');
    boxplot([results.SSIM_Input, results.SSIM_CNN], 'Labels', {'Entrada', 'CNN'});
    ylabel('SSIM'); title('SSIM respecto a referencia CLAHE');
    exportgraphics(f2, fullfile(resultsFolder, 'boxplot_ssim.png')); close(f2);

    f3 = figure('Visible', 'off', 'Color', 'w');
    boxplot([results.CNR_Input, results.CNR_CNN], 'Labels', {'Entrada', 'CNN'});
    ylabel('CNR'); title('Contraste ruido (CNR)');
    exportgraphics(f3, fullfile(resultsFolder, 'boxplot_cnr.png')); close(f3);

    f4 = figure('Visible', 'off', 'Color', 'w');
    histogram(results.Delta_PSNR, 15);
    xlabel('\Delta PSNR'); ylabel('Frecuencia');
    title('Distribución de mejora PSNR (CNN - Entrada)');
    exportgraphics(f4, fullfile(resultsFolder, 'hist_delta_psnr.png')); close(f4);
end

function analyzeNetworkIfPossible(layers)
    try
        analyzeNetwork(layers);
    catch
        disp('analyzeNetwork no disponible; continuando...');
    end
end

function savejsonIfPossible(outPath, S)
    try
        txt = jsonencode(S, 'PrettyPrint', true);
        fid = fopen(outPath, 'w');
        fwrite(fid, txt, 'char');
        fclose(fid);
    catch
        disp('No se pudo guardar JSON.');
    end
end
