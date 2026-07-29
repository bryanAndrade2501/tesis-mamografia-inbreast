function results = evaluatePatientsAtNativeResolution( ...
        net, cfg, filePaths, fileNames, patientIds, targetPatientIds)
%EVALUATEPATIENTSATNATIVERESOLUTION Inferencia y métricas a tamaño nativo (tras ROI).
%   Procesa una imagen a la vez para evitar tensores gigantes en memoria.
%   La red debe estar entrenada (típicamente a 512x512); predict usa la CNN
%   de forma fully-convolutional sobre el tamaño nativo.

    if nargin < 6 || isempty(targetPatientIds)
        targetPatientIds = {};
    end
    targetPatientIds = cellstr(string(targetPatientIds));

    idxList = find(ismember(patientIds, targetPatientIds));
    if isempty(idxList)
        warning('No se encontraron imágenes para pacientes: %s', strjoin(targetPatientIds, ', '));
        results = struct('perImage', table(), 'summary', table());
        return;
    end

    cfgNative = cfg;
    cfgNative.keepOriginalSize = true;

    n = numel(idxList);
    rows = cell(n, 1);

    for k = 1:n
        i = idxList(k);
        fprintf('  Full-res %d/%d: %s (%s)\n', k, n, fileNames{i}, patientIds{i});

        [Iraw, ~] = readDicomMammography(filePaths{i});
        Iprep = preprocessMammography(Iraw, cfgNative, struct());
        Iden = applyDenoising(Iprep, cfgNative);
        Yref = applyCLAHE(Iden, cfgNative);

        Xin = im2single(Iden);
        X4 = reshape(Xin, [size(Xin) 1 1]);

        try
            Y4 = predict(net, X4, 'ExecutionEnvironment', cfg.executionEnvironment);
            Ypred = clip01(squeeze(Y4));
        catch ME
            error('predict falló en tamaño %dx%d: %s', size(Xin, 1), size(Xin, 2), ME.message);
        end

        if ~isequal(size(Ypred), size(Yref))
            Ypred = imresize(Ypred, size(Yref), 'bilinear');
        end

        row = struct();
        row.Imagen = string(fileNames{i});
        row.Paciente = string(patientIds{i});
        row.Filas = size(Xin, 1);
        row.Columnas = size(Xin, 2);
        row.PSNR_Input = psnr(Xin, Yref);
        row.PSNR_CNN = psnr(Ypred, Yref);
        row.SSIM_Input = ssim(Xin, Yref);
        row.SSIM_CNN = ssim(Ypred, Yref);
        row.MAE_Input = mean(abs(Xin(:) - Yref(:)), 'omitnan');
        row.MAE_CNN = mean(abs(Ypred(:) - Yref(:)), 'omitnan');
        row.CNR_Input = computeAutomaticCNR(Xin, cfgNative);
        row.CNR_CNN = computeAutomaticCNR(Ypred, cfgNative);
        row.Delta_PSNR = row.PSNR_CNN - row.PSNR_Input;
        row.Delta_SSIM = row.SSIM_CNN - row.SSIM_Input;
        row.Delta_CNR = row.CNR_CNN - row.CNR_Input;
        rows{k} = row;

        saveFullResPreview(Xin, Ypred, Yref, fileNames{i}, cfgNative);
        saveFullResArrays(Xin, Ypred, Yref, fileNames{i}, cfgNative);
    end

    perImage = struct2table([rows{:}]);
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

%% --- Subfunciones (mismo pipeline que tesis_enhancement_inbreast.m) ---

function [I, meta] = readDicomMammography(filePath)
    info = struct();
    try
        info = dicominfo(filePath);
    catch
    end
    I = double(dicomread(filePath));
    if isfield(info, 'RescaleSlope'), slope = double(info.RescaleSlope); else, slope = 1; end
    if isfield(info, 'RescaleIntercept'), intercept = double(info.RescaleIntercept); else, intercept = 0; end
    I = I * slope + intercept;
    if isfield(info, 'PhotometricInterpretation') && strcmpi(info.PhotometricInterpretation, 'MONOCHROME1')
        I = max(I(:)) - I;
    end
    meta = struct('filePath', filePath);
end

function Iout = preprocessMammography(I, cfg, ~)
    I = double(I);
    I(I < 0) = 0;
    if cfg.cropBreastROI
        I = cropBreastBoundingBox(I, cfg.roiMarginPx);
    end
    if isfield(cfg, 'keepOriginalSize') && ~cfg.keepOriginalSize
        I = imresize(I, cfg.imageSize, 'bilinear');
    end
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
    if nnz(mask) < 100, Icrop = I; return; end
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
        case 'median',  Iout = medfilt2(I, cfg.medianKernel);
        case 'bilateral', Iout = imbilatfilt(I, cfg.bilateralDegree, cfg.bilateralSpatial);
        case 'none',    Iout = I;
        otherwise, error('Método de denoise no soportado: %s', cfg.denoiseMethod);
    end
    Iout = min(max(Iout, 0), 1);
end

function Iout = applyCLAHE(I, cfg)
    Iout = adapthisteq(I, 'NumTiles', cfg.claheNumTiles, ...
        'ClipLimit', cfg.claheClipLimit, 'NBins', cfg.claheNBins);
    Iout = min(max(im2single(Iout), 0), 1);
end

function Iout = clip01(I)
    Iout = min(max(I, 0), 1);
end

function cnrValue = computeAutomaticCNR(I, cfg)
    I = mat2gray(I);
    breastMask = I > graythresh(I) * 0.25;
    breastMask = bwareaopen(breastMask, cfg.minMaskArea);
    breastMask = imfill(breastMask, 'holes');
    if nnz(breastMask) < 100, cnrValue = NaN; return; end
    breastPixels = I(breastMask);
    th = graythresh(breastPixels) * cfg.otsuScaleForeground;
    lesionMask = false(size(I));
    lesionMask(breastMask) = I(breastMask) > th;
    lesionMask = bwareaopen(lesionMask, cfg.minMaskArea);
    backgroundMask = breastMask & ~lesionMask;
    if nnz(lesionMask) < 50 || nnz(backgroundMask) < 50, cnrValue = NaN; return; end
    mu1 = mean(I(lesionMask), 'omitnan');
    mu0 = mean(I(backgroundMask), 'omitnan');
    s1 = std(I(lesionMask), 0, 'omitnan');
    s0 = std(I(backgroundMask), 0, 'omitnan');
    denom = sqrt(s1^2 + s0^2);
    if denom < eps, cnrValue = NaN; else, cnrValue = abs(mu1 - mu0) / denom; end
end

function saveFullResPreview(Iin, Iprd, Itgt, fileName, cfg)
    previewFolder = fullfile(cfg.resultsFolder, 'previews');
    if ~exist(previewFolder, 'dir'), mkdir(previewFolder); end
    [~, base, ~] = fileparts(fileName);
    f = figure('Visible', 'off', 'Color', 'w', 'Position', [80 80 1500 420]);
    subplot(1, 3, 1); imshow(Iin, []);  title('Entrada (full-res)');
    subplot(1, 3, 2); imshow(Iprd, []);  title('Salida CNN (full-res)');
    subplot(1, 3, 3); imshow(Itgt, []);  title('Referencia CLAHE');
    sgtitle(sprintf('Full-res: %s', fileName), 'Interpreter', 'none');
    exportgraphics(f, fullfile(previewFolder, ['preview_fullres_' base '.png']));
    close(f);
end

function saveFullResArrays(Iin, Iprd, Itgt, fileName, cfg)
    matFolder = fullfile(cfg.resultsFolder, 'arrays');
    if ~exist(matFolder, 'dir'), mkdir(matFolder); end
    [~, base, ~] = fileparts(fileName);
    save(fullfile(matFolder, ['fullres_' base '.mat']), 'Iin', 'Iprd', 'Itgt', '-v7.3');
end
