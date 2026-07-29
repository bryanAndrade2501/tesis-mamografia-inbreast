%% Inferencia a resolución nativa (sin reentrenar)
% Usa la red ya entrenada a 512x512 y evalúa pacientes específicos
% a tamaño original (tras ROI).
%
% Requisito: haber ejecutado tesis_enhancement_inbreast.m al menos una vez
% y tener red_enhancement_inbreast.mat en resultados_inbreast_enhancement/

clear; clc;

resultsFolder = fullfile(pwd, 'resultados_inbreast_enhancement');
modelPath = fullfile(resultsFolder, 'red_enhancement_inbreast.mat');

assert(isfile(modelPath), ...
    'No se encontró el modelo entrenado: %s\nEjecuta primero tesis_enhancement_inbreast.m', modelPath);

S = load(modelPath, 'net', 'cfg');
net = S.net;
cfg = S.cfg;

% Pacientes a evaluar en full-res (por defecto 20588680)
if ~isfield(cfg, 'fullResTestPatientIds') || isempty(cfg.fullResTestPatientIds)
    cfg.fullResTestPatientIds = {'20588680'};
end

fullResFolder = fullfile(resultsFolder, 'fullres');
if ~exist(fullResFolder, 'dir')
    mkdir(fullResFolder);
end
cfg.resultsFolder = fullResFolder;
cfg.keepOriginalSize = true;

fprintf('=== Inferencia full-res | pacientes: %s ===\n', strjoin(cfg.fullResTestPatientIds, ', '));

dicomFiles = dir(fullfile(cfg.dicomFolder, '**', '*.dcm'));
assert(~isempty(dicomFiles), 'No hay archivos DICOM en: %s', cfg.dicomFolder);

filePaths = arrayfun(@(f) fullfile(f.folder, f.name), dicomFiles, 'UniformOutput', false);
fileNames = {dicomFiles.name}';
patientIds = cell(numel(fileNames), 1);
for i = 1:numel(fileNames)
    parts = split(fileNames{i}, '_');
    patientIds{i} = char(parts(1));
end

fullResResults = evaluatePatientsAtNativeResolution( ...
    net, cfg, filePaths, fileNames, patientIds, cfg.fullResTestPatientIds);

writetable(fullResResults.perImage, fullfile(fullResFolder, 'metricas_test_fullres.csv'));
writetable(fullResResults.summary, fullfile(fullResFolder, 'resumen_metricas_fullres.csv'));

disp(fullResResults.summary);
fprintf('\nResultados full-res en: %s\n', fullResFolder);
