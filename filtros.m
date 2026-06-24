%% filtros.m - Comparación de métodos de reducción de ruido (INbreast)
% Prueba en N imágenes aleatorias:
%   none | median | bilateral | gaussian | laplacian
%
% Métricas principales (propuesta práctica):
%   CNR_CLAHE, Std_CLAHE, SSIM_CLAHE_vs_ref, MAE_CLAHE_vs_ref
%
% Salidas:
%   - Figuras review: original + resultado de cada filtro
%   - Documento HTML/TXT con métricas por imagen
% Sección tesis: 3.4.3 Reducción de ruido

clear; clc; close all;

%% ========================================================================
%  CONFIGURACIÓN
% ========================================================================
cfg = struct();

cfg.inbreastRoot  = 'C:\Users\Asus\Documents\Tesis\imagenes\INbreast Release 1.0';
cfg.dicomFolder   = fullfile(cfg.inbreastRoot, 'AllDICOMs');
cfg.resultsFolder = fullfile(pwd, 'resultados_comparacion_filtros');
cfg.imageSize     = [512 512];
cfg.randomSeed    = 42;
cfg.numSamples    = 10;

cfg.cropBreastROI      = true;
cfg.roiMarginPx        = 8;
cfg.normLowPercentile  = 0.5;
cfg.normHighPercentile = 99.5;

cfg.medianKernel       = [3 3];
cfg.bilateralDegree    = 3;
cfg.bilateralSpatial   = 3;
cfg.gaussianSigma      = 1.0;
cfg.laplacianSigma     = 1.2;
cfg.laplacianAlpha     = 0.40;

cfg.claheNumTiles      = [8 8];
cfg.claheClipLimit     = 0.01;
cfg.claheNBins         = 256;

cfg.otsuScaleForeground = 1.00;
cfg.minMaskArea         = 200;

filterNames  = {'none', 'median', 'bilateral', 'gaussian', 'laplacian'};
filterLabels = {'Sin filtro', 'Mediana 3x3', 'Bilateral', 'Gaussiano', 'Laplaciano'};

rng(cfg.randomSeed);

if ~exist(cfg.resultsFolder, 'dir'), mkdir(cfg.resultsFolder); end
previewFolder = fullfile(cfg.resultsFolder, 'previews');
reviewFolder  = fullfile(cfg.resultsFolder, 'review');
if ~exist(previewFolder, 'dir'), mkdir(previewFolder); end
if ~exist(reviewFolder, 'dir'), mkdir(reviewFolder); end

fprintf('=== Comparación de filtros de reducción de ruido ===\n');

%% ========================================================================
%  CARGAR LISTA DE DICOM
% ========================================================================
dicomFiles = dir(fullfile(cfg.dicomFolder, '**', '*.dcm'));
assert(~isempty(dicomFiles), 'No se encontraron archivos .dcm en: %s', cfg.dicomFolder);

filePaths = arrayfun(@(f) fullfile(f.folder, f.name), dicomFiles, 'UniformOutput', false);
fileNames = {dicomFiles.name}';
numImages = numel(filePaths);

sampleCount = min(cfg.numSamples, numImages);
sampleIdx = randperm(numImages, sampleCount);
sampleNames = fileNames(sampleIdx);

fprintf('Imágenes totales: %d | Muestra aleatoria: %d\n', numImages, sampleCount);

%% ========================================================================
%  PROCESAR MUESTRA
% ========================================================================
nFilters = numel(filterNames);
metricRows = [];
reviewCache = cell(sampleCount, 1);

for s = 1:sampleCount
    fprintf('Procesando muestra %d / %d: %s\n', s, sampleCount, sampleNames{s});

    [Iraw, ~] = readDicomMammography(filePaths{sampleIdx(s)});
    Iprep = preprocessMammography(Iraw, cfg);
    IclaheRef = applyCLAHE(Iprep, cfg);

    denoisedImages = cell(1, nFilters);
    claheImages    = cell(1, nFilters);
    imageMetrics   = table();

    for f = 1:nFilters
        cfg.denoiseMethod = filterNames{f};
        Iden = applyDenoising(Iprep, cfg);
        Icl  = applyCLAHE(Iden, cfg);

        denoisedImages{f} = Iden;
        claheImages{f}    = Icl;

        m = computeKeyMetrics(Icl, IclaheRef, cfg);

        row = table( ...
            string(sampleNames{s}), string(filterLabels{f}), string(filterNames{f}), ...
            m.CNR_CLAHE, m.Std_CLAHE, m.SSIM_CLAHE_vs_ref, m.MAE_CLAHE_vs_ref, ...
            'VariableNames', {'Imagen', 'Filtro', 'FiltroKey', ...
            'CNR_CLAHE', 'Std_CLAHE', 'SSIM_CLAHE_vs_ref', 'MAE_CLAHE_vs_ref'});

        row.ScoreFiltro = rankScoreSingle(row.CNR_CLAHE, row.Std_CLAHE, ...
            row.SSIM_CLAHE_vs_ref, row.MAE_CLAHE_vs_ref);
        imageMetrics = [imageMetrics; row]; %#ok<AGROW>
    end

    [~, bestIdx] = min(imageMetrics.ScoreFiltro);
    imageMetrics.EsMejorEnImagen = false(height(imageMetrics), 1);
    imageMetrics.EsMejorEnImagen(bestIdx) = true;
    metricRows = [metricRows; imageMetrics]; %#ok<AGROW>

    reviewCache{s} = struct( ...
        'index', s, ...
        'name', sampleNames{s}, ...
        'Iprep', Iprep, ...
        'denoisedImages', {denoisedImages}, ...
        'claheImages', {claheImages}, ...
        'metrics', imageMetrics, ...
        'bestFilter', imageMetrics.Filtro{bestIdx});

    saveFilterComparisonFigure(Iprep, denoisedImages, claheImages, ...
        filterLabels, sampleNames{s}, previewFolder, s);

    saveReviewPanel(Iprep, claheImages, filterLabels, imageMetrics, ...
        sampleNames{s}, reviewFolder, s);
end

metricsTable = metricRows;
writetable(metricsTable, fullfile(cfg.resultsFolder, 'metricas_por_imagen_filtro.csv'));

%% ========================================================================
%  RESUMEN Y RANKING GLOBAL
% ========================================================================
summaryTable = groupsummary(metricsTable, 'Filtro', {'mean', 'std'}, ...
    {'CNR_CLAHE', 'Std_CLAHE', 'SSIM_CLAHE_vs_ref', 'MAE_CLAHE_vs_ref'});

writetable(summaryTable, fullfile(cfg.resultsFolder, 'resumen_filtros.csv'));

rankCNR  = scoreByColumn(summaryTable, 'mean_CNR_CLAHE', true);
rankStd  = scoreByColumn(summaryTable, 'mean_Std_CLAHE', false);
rankSSIM = scoreByColumn(summaryTable, 'mean_SSIM_CLAHE_vs_ref', true);
rankMAE  = scoreByColumn(summaryTable, 'mean_MAE_CLAHE_vs_ref', false);

rankingTable = table(summaryTable.Filtro, rankCNR, rankStd, rankSSIM, rankMAE, ...
    'VariableNames', {'Filtro', 'Rank_CNR', 'Rank_StdRuido', 'Rank_SSIM', 'Rank_MAE'});
rankingTable.ScoreTotal = rankCNR + rankStd + rankSSIM + rankMAE;
rankingTable = sortrows(rankingTable, 'ScoreTotal', 'ascend');

writetable(rankingTable, fullfile(cfg.resultsFolder, 'ranking_filtros.csv'));

bestFilter = rankingTable.Filtro{1};

%% ========================================================================
%  REVIEW FINAL + DOCUMENTO DE MÉTRICAS
% ========================================================================
fprintf('\nGenerando review final y documento de métricas...\n');

generateFinalReviewMosaic(reviewCache, filterLabels, reviewFolder, cfg);
generateMetricsDocument(metricsTable, summaryTable, rankingTable, reviewCache, cfg);

fprintf('\n--- Resumen global (4 métricas clave) ---\n');
disp(summaryTable(:, {'Filtro', 'mean_CNR_CLAHE', 'mean_Std_CLAHE', ...
    'mean_SSIM_CLAHE_vs_ref', 'mean_MAE_CLAHE_vs_ref'}));
fprintf('\nFiltro recomendado (menor ScoreTotal): %s\n', bestFilter);
fprintf('Resultados en: %s\n', cfg.resultsFolder);
fprintf('  - review/review_muestra_XX.png\n');
fprintf('  - review/REPORTE_METRICAS_FILTROS.html\n');
fprintf('  - review/REPORTE_METRICAS_FILTROS.txt\n');
fprintf('  - metricas_por_imagen_filtro.csv\n');

makeSummaryPlots(metricsTable, summaryTable, cfg.resultsFolder, sampleCount);

%% ========================================================================
%  FUNCIONES LOCALES
% ========================================================================

function m = computeKeyMetrics(Icl, IclaheRef, cfg)
    m = struct();
    m.CNR_CLAHE          = computeAutomaticCNR(Icl, cfg);
    m.Std_CLAHE          = estimateNoiseStd(Icl);
    m.SSIM_CLAHE_vs_ref  = ssim(Icl, IclaheRef);
    m.MAE_CLAHE_vs_ref   = mean(abs(Icl(:) - IclaheRef(:)), 'omitnan');
end

function score = rankScoreSingle(cnr, stdv, ssimVal, mae)
    % Puntuación heurística por imagen (menor = mejor balance)
    score = 0;
    score = score + max(0, 2.5 - cnr);
    score = score + stdv * 10;
    score = score + max(0, 1 - ssimVal) * 5;
    score = score + mae * 10;
end

function saveReviewPanel(Iprep, claheImages, labels, imageMetrics, imageName, outFolder, idx)
    nFilters = numel(labels);
    nCols = nFilters + 1;

    f = figure('Visible', 'off', 'Color', 'w', 'Position', [20 20 2000 420]);

    subplot(1, nCols, 1);
    imshow(Iprep, []);
    title({'Original', '(preprocesada)'}, 'FontSize', 9);

    for c = 1:nFilters
        subplot(1, nCols, c + 1);
        imshow(claheImages{c}, []);

        met = imageMetrics(c, :);
        titleStr = sprintf('%s\nCNR=%.2f SSIM=%.3f', labels{c}, met.CNR_CLAHE, met.SSIM_CLAHE_vs_ref);
        if met.EsMejorEnImagen
            title({[labels{c} ' *'], titleStr}, 'FontSize', 9, 'Color', [0 0.5 0]);
        else
            title({labels{c}, titleStr}, 'FontSize', 9);
        end
    end

    sgtitle(sprintf('Review muestra %02d | %s  (* = mejor filtro en esta imagen)', ...
        idx, imageName), 'Interpreter', 'none', 'FontSize', 10);
    exportgraphics(f, fullfile(outFolder, sprintf('review_muestra_%02d.png', idx)));
    close(f);
end

function generateFinalReviewMosaic(reviewCache, filterLabels, reviewFolder, cfg)
    sampleCount = numel(reviewCache);
    nFilters = numel(filterLabels);
    nCols = nFilters + 1;

    f = figure('Visible', 'off', 'Color', 'w', 'Position', [10 10 2000 200 * sampleCount]);

    for s = 1:sampleCount
        data = reviewCache{s};
        for c = 0:nFilters
            axIdx = (s - 1) * nCols + c + 1;
            subplot(sampleCount, nCols, axIdx);
            if c == 0
                imshow(data.Iprep, []);
                if s == 1, title('Original', 'FontSize', 8); end
            else
                imshow(data.claheImages{c}, []);
                if s == 1
                    title(filterLabels{c}, 'FontSize', 8);
                end
            end
            if c == 0
                ylabel(sprintf('#%02d', s), 'FontWeight', 'bold');
            end
            axis off;
        end
    end

    sgtitle('Review final: original vs filtros (denoise + CLAHE)', 'FontSize', 12);
    exportgraphics(f, fullfile(reviewFolder, 'REVIEW_FINAL_TODAS_MUESTRAS.png'), 'Resolution', 120);
    close(f);
end

function generateMetricsDocument(metricsTable, summaryTable, rankingTable, reviewCache, cfg)
    outHtml = fullfile(cfg.resultsFolder, 'review', 'REPORTE_METRICAS_FILTROS.html');
    outTxt  = fullfile(cfg.resultsFolder, 'review', 'REPORTE_METRICAS_FILTROS.txt');

    fid = fopen(outTxt, 'w', 'n', 'UTF-8');
    fprintf(fid, 'REPORTE DE COMPARACION DE FILTROS - INbreast\n');
    fprintf(fid, 'Fecha: %s\n', datestr(now));
    fprintf(fid, 'Muestras: %d | Semilla: %d\n\n', numel(reviewCache), cfg.randomSeed);
    fprintf(fid, 'METRICAS CLAVE (propuesta practica):\n');
    fprintf(fid, '  CNR_CLAHE          -> mas alto es mejor\n');
    fprintf(fid, '  Std_CLAHE          -> mas bajo es mejor\n');
    fprintf(fid, '  SSIM_CLAHE_vs_ref  -> mas alto es mejor\n');
    fprintf(fid, '  MAE_CLAHE_vs_ref   -> mas bajo es mejor\n\n');

    fprintf(fid, '=== RANKING GLOBAL ===\n');
    for i = 1:height(rankingTable)
        fprintf(fid, '%d. %s (ScoreTotal=%d)\n', i, rankingTable.Filtro{i}, rankingTable.ScoreTotal(i));
    end
    fprintf(fid, '\n=== RESUMEN POR FILTRO (media) ===\n');
    for i = 1:height(summaryTable)
        fprintf(fid, '\n%s:\n', summaryTable.Filtro{i});
        fprintf(fid, '  CNR_CLAHE         = %.4f (std=%.4f)\n', ...
            summaryTable.mean_CNR_CLAHE(i), summaryTable.std_CNR_CLAHE(i));
        fprintf(fid, '  Std_CLAHE         = %.4f (std=%.4f)\n', ...
            summaryTable.mean_Std_CLAHE(i), summaryTable.std_Std_CLAHE(i));
        fprintf(fid, '  SSIM_CLAHE_vs_ref = %.4f (std=%.4f)\n', ...
            summaryTable.mean_SSIM_CLAHE_vs_ref(i), summaryTable.std_SSIM_CLAHE_vs_ref(i));
        fprintf(fid, '  MAE_CLAHE_vs_ref  = %.4f (std=%.4f)\n', ...
            summaryTable.mean_MAE_CLAHE_vs_ref(i), summaryTable.std_MAE_CLAHE_vs_ref(i));
    end

    fprintf(fid, '\n=== METRICAS POR IMAGEN ===\n');
    images = unique(metricsTable.Imagen, 'stable');
    for i = 1:numel(images)
        imgName = images(i);
        fprintf(fid, '\n--- %s ---\n', imgName);
        sub = metricsTable(metricsTable.Imagen == imgName, :);
        for j = 1:height(sub)
            bestMark = '';
            if ismember('EsMejorEnImagen', sub.Properties.VariableNames)
                if sub.EsMejorEnImagen(j), bestMark = ' [MEJOR]'; end
            end
            fprintf(fid, '%s%s\n', sub.Filtro(j), bestMark);
            fprintf(fid, '  CNR=%.4f | Std=%.4f | SSIM=%.4f | MAE=%.4f\n', ...
                sub.CNR_CLAHE(j), sub.Std_CLAHE(j), sub.SSIM_CLAHE_vs_ref(j), sub.MAE_CLAHE_vs_ref(j));
        end
        fprintf(fid, 'Review visual: review/review_muestra_%02d.png\n', i);
    end
    fclose(fid);

    html = ['<!DOCTYPE html><html><head><meta charset="UTF-8">' ...
        '<title>Reporte comparación filtros</title>' ...
        '<style>body{font-family:Arial,sans-serif;margin:24px;}' ...
        'table{border-collapse:collapse;width:100%;margin:12px 0;}' ...
        'th,td{border:1px solid #ccc;padding:8px;text-align:center;}' ...
        'th{background:#f0f0f0;} .best{background:#e8f5e9;font-weight:bold;}' ...
        'h2{margin-top:28px;}</style></head><body>'];

    html = [html '<h1>Reporte de comparación de filtros (INbreast)</h1>'];
    html = [html '<p><b>Fecha:</b> ' datestr(now) '</p>'];
    html = [html '<p><b>Muestras:</b> ' num2str(numel(reviewCache)) ...
        ' | <b>Semilla:</b> ' num2str(cfg.randomSeed) '</p>'];

    html = [html '<h2>Métricas clave</h2><ul>'];
    html = [html '<li><b>CNR_CLAHE</b> — más alto es mejor (contraste)</li>'];
    html = [html '<li><b>Std_CLAHE</b> — más bajo es mejor (ruido)</li>'];
    html = [html '<li><b>SSIM_CLAHE_vs_ref</b> — más alto es mejor (estructura)</li>'];
    html = [html '<li><b>MAE_CLAHE_vs_ref</b> — más bajo es mejor (error píxel)</li></ul>'];

    html = [html '<h2>Ranking global</h2><table><tr><th>#</th><th>Filtro</th><th>ScoreTotal</th></tr>'];
    for i = 1:height(rankingTable)
        html = [html '<tr><td>' num2str(i) '</td><td>' char(rankingTable.Filtro(i)) ...
            '</td><td>' num2str(rankingTable.ScoreTotal(i)) '</td></tr>'];
    end
    html = [html '</table>'];

    html = [html '<h2>Resumen por filtro</h2><table>'];
    html = [html '<tr><th>Filtro</th><th>CNR (media±std)</th><th>Std (media±std)</th>' ...
        '<th>SSIM (media±std)</th><th>MAE (media±std)</th></tr>'];
    for i = 1:height(summaryTable)
        html = [html '<tr><td>' char(summaryTable.Filtro(i)) '</td>'];
        html = [html '<td>' sprintf('%.3f ± %.3f', summaryTable.mean_CNR_CLAHE(i), summaryTable.std_CNR_CLAHE(i)) '</td>'];
        html = [html '<td>' sprintf('%.4f ± %.4f', summaryTable.mean_Std_CLAHE(i), summaryTable.std_Std_CLAHE(i)) '</td>'];
        html = [html '<td>' sprintf('%.3f ± %.3f', summaryTable.mean_SSIM_CLAHE_vs_ref(i), summaryTable.std_SSIM_CLAHE_vs_ref(i)) '</td>'];
        html = [html '<td>' sprintf('%.4f ± %.4f', summaryTable.mean_MAE_CLAHE_vs_ref(i), summaryTable.std_MAE_CLAHE_vs_ref(i)) '</td></tr>'];
    end
    html = [html '</table>'];

    for i = 1:numel(reviewCache)
        data = reviewCache{i};
        imgName = char(data.name);
        html = [html '<h2>Muestra ' sprintf('%02d', i) ': ' imgName '</h2>'];
        html = [html '<img src="review_muestra_' sprintf('%02d', i) '.png" width="100%" alt="review"/>'];
        html = [html '<table><tr><th>Filtro</th><th>CNR</th><th>Std</th><th>SSIM</th><th>MAE</th></tr>'];

        sub = data.metrics;
        for j = 1:height(sub)
            cls = '';
            if sub.EsMejorEnImagen(j), cls = ' class="best"'; end
            html = [html '<tr' cls '><td>' char(sub.Filtro(j)) '</td>'];
            html = [html '<td>' sprintf('%.4f', sub.CNR_CLAHE(j)) '</td>'];
            html = [html '<td>' sprintf('%.4f', sub.Std_CLAHE(j)) '</td>'];
            html = [html '<td>' sprintf('%.4f', sub.SSIM_CLAHE_vs_ref(j)) '</td>'];
            html = [html '<td>' sprintf('%.4f', sub.MAE_CLAHE_vs_ref(j)) '</td></tr>'];
        end
        html = [html '</table>'];
    end

    html = [html '<h2>Review completo</h2>'];
    html = [html '<img src="REVIEW_FINAL_TODAS_MUESTRAS.png" width="100%" alt="review final"/>'];
    html = [html '</body></html>'];

    fid = fopen(outHtml, 'w', 'n', 'UTF-8');
    fwrite(fid, html, 'char');
    fclose(fid);
end

function Iden = applyDenoising(I, cfg)
    switch lower(cfg.denoiseMethod)
        case 'none'
            Iden = I;
        case 'median'
            Iden = medfilt2(I, cfg.medianKernel);
        case 'bilateral'
            Iden = imbilatfilt(I, cfg.bilateralDegree, cfg.bilateralSpatial);
        case 'gaussian'
            Iden = imgaussfilt(I, cfg.gaussianSigma);
        case 'laplacian'
            Iden = applyLaplacianDenoise(I, cfg.laplacianSigma, cfg.laplacianAlpha);
        otherwise
            error('Filtro no soportado: %s', cfg.denoiseMethod);
    end
    Iden = min(max(Iden, 0), 1);
end

function Iout = applyLaplacianDenoise(I, sigma, alpha)
    lowPass = imgaussfilt(I, sigma);
    highPass = I - lowPass;
    Iout = lowPass + alpha * highPass;
end

function Iout = applyCLAHE(I, cfg)
    Iout = adapthisteq(I, ...
        'NumTiles', cfg.claheNumTiles, ...
        'ClipLimit', cfg.claheClipLimit, ...
        'NBins', cfg.claheNBins);
    Iout = min(max(im2single(Iout), 0), 1);
end

function [I, meta] = readDicomMammography(filePath)
    info = struct();
    try, info = dicominfo(filePath); catch, end
    I = double(dicomread(filePath));
    slope = 1; intercept = 0;
    if isfield(info, 'RescaleSlope'), slope = double(info.RescaleSlope); end
    if isfield(info, 'RescaleIntercept'), intercept = double(info.RescaleIntercept); end
    I = I * slope + intercept;
    if isfield(info, 'PhotometricInterpretation') && ...
            strcmpi(info.PhotometricInterpretation, 'MONOCHROME1')
        I = max(I(:)) - I;
    end
    meta = struct('filePath', filePath);
end

function Iout = preprocessMammography(I, cfg)
    I = double(I);
    I(I < 0) = 0;
    if cfg.cropBreastROI, I = cropBreastBoundingBox(I, cfg.roiMarginPx); end
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
    if nnz(mask) < 100, Icrop = I; return; end
    stats = regionprops(mask, 'BoundingBox');
    [~, idx] = max(cellfun(@(bb) bb(3) * bb(4), {stats.BoundingBox}));
    bb = stats(idx).BoundingBox;
    r1 = max(1, floor(bb(2)) - marginPx);
    c1 = max(1, floor(bb(1)) - marginPx);
    r2 = min(size(I, 1), ceil(bb(2) + bb(4)) + marginPx);
    c2 = min(size(I, 2), ceil(bb(1) + bb(3)) + marginPx);
    Icrop = I(r1:r2, c1:c2);
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

function noiseStd = estimateNoiseStd(I)
    I = mat2gray(I);
    mask = I > graythresh(I) * 0.25;
    mask = bwareaopen(mask, 200);
    mask = imfill(mask, 'holes');
    if nnz(mask) < 100, noiseStd = NaN; return; end
    vals = I(mask);
    ref = vals(vals < prctile(vals, 40));
    if numel(ref) < 50, ref = vals; end
    noiseStd = std(ref, 0, 'omitnan');
end

function saveFilterComparisonFigure(Iprep, denoisedImages, claheImages, labels, imageName, outFolder, idx)
    nFilters = numel(labels);
    f = figure('Visible', 'off', 'Color', 'w', 'Position', [40 40 1600 900]);
    for c = 1:nFilters
        subplot(2, nFilters, c);
        imshow(denoisedImages{c}, []);
        title(sprintf('%s\n(denoise)', labels{c}), 'FontSize', 9);
        subplot(2, nFilters, nFilters + c);
        imshow(claheImages{c}, []);
        title(sprintf('%s\n(+ CLAHE)', labels{c}), 'FontSize', 9);
    end
    sgtitle(sprintf('Detalle muestra %02d | %s', idx, imageName), 'Interpreter', 'none');
    exportgraphics(f, fullfile(outFolder, sprintf('comparacion_filtros_%02d.png', idx)));
    close(f);
end

function ranks = scoreByColumn(T, colName, higherIsBetter)
    values = T.(colName);
    if higherIsBetter, [~, order] = sort(values, 'descend');
    else, [~, order] = sort(values, 'ascend'); end
    ranks = zeros(height(T), 1);
    ranks(order) = 1:height(T);
end

function makeSummaryPlots(metricsTable, summaryTable, resultsFolder, sampleCount)
    filtros = summaryTable.Filtro;

    f1 = figure('Visible', 'off', 'Color', 'w');
    bar(categorical(filtros), summaryTable.mean_CNR_CLAHE);
    ylabel('CNR'); title(sprintf('CNR por filtro (n=%d)', sampleCount));
    exportgraphics(f1, fullfile(resultsFolder, 'bar_cnr_filtros.png')); close(f1);

    f2 = figure('Visible', 'off', 'Color', 'w');
    bar(categorical(filtros), summaryTable.mean_Std_CLAHE);
    ylabel('Std ruido'); title('Ruido residual (menor es mejor)');
    exportgraphics(f2, fullfile(resultsFolder, 'bar_std_filtros.png')); close(f2);

    f3 = figure('Visible', 'off', 'Color', 'w');
    bar(categorical(filtros), summaryTable.mean_SSIM_CLAHE_vs_ref);
    ylabel('SSIM'); title('SSIM vs referencia CLAHE');
    exportgraphics(f3, fullfile(resultsFolder, 'bar_ssim_filtros.png')); close(f3);

    f4 = figure('Visible', 'off', 'Color', 'w');
    bar(categorical(filtros), summaryTable.mean_MAE_CLAHE_vs_ref);
    ylabel('MAE'); title('MAE vs referencia CLAHE (menor es mejor)');
    exportgraphics(f4, fullfile(resultsFolder, 'bar_mae_filtros.png')); close(f4);
end
