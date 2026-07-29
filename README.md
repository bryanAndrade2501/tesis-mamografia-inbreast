# Mejora de imágenes mamográficas DICOM (INbreast) con CNN

Pipeline en MATLAB para la tesis: adquisición DICOM, preprocesamiento, reducción de ruido, mejora de contraste (CLAHE) y entrenamiento/evaluación de una CNN tipo U-Net.

## Requisitos

- MATLAB R2020b o superior (recomendado)
- **Deep Learning Toolbox**
- **Image Processing Toolbox**
- GPU opcional (acelera el entrenamiento)

## Dataset

Descargar **INbreast Release 1.0** (Kaggle o repositorio oficial) y descomprimir. La estructura esperada es:

```
INbreast Release 1.0/
├── AllDICOMs/      ← archivos .dcm
├── INbreast.csv
└── ...
```

## Configuración

1. Clonar este repositorio.
2. Abrir `tesis_enhancement_inbreast.m` en MATLAB.
3. Editar la ruta del dataset:

```matlab
cfg.inbreastRoot = 'RUTA\A\TU\INbreast Release 1.0';
```

4. Ejecutar el script completo en MATLAB:

```matlab
tesis_enhancement_inbreast
```

## Salidas

Los resultados se guardan en `resultados_inbreast_enhancement/`:

- `metricas_test.csv` — métricas por imagen (PSNR, SSIM, MAE, CNR)
- `resumen_metricas.csv` — promedios
- `previews/` — comparativas visuales
- `red_enhancement_inbreast.mat` — red entrenada
- `fullres/` — evaluación a tamaño nativo para pacientes en `fullResTestPatientIds`

### Inferencia full-res sin reentrenar

```matlab
infer_single_fullres
```

Evalúa pacientes como `20588680` a tamaño original (una imagen a la vez, bajo uso de RAM).

### Exportar salidas CNN como DICOM

Tras entrenar el modelo, ejecutar:

```matlab
export_cnn_dicom
```

Genera **410 archivos DICOM por carpeta** en `../results_images/`:

| Carpeta | Contenido | Sufijo |
|---------|-----------|--------|
| `entrada/` | X — preprocesamiento + denoise | `_INPUT.dcm` |
| `referencia_clahe/` | Y — objetivo CLAHE | `_CLAHE.dcm` |
| `cnn/` | Salida de la U-Net | `_CNN.dcm` |

Usa el modelo de `RESULTADOS/EX2/` y `dataset_cache.mat`. Incluye `export_log.csv` con el mapeo de los tres tipos.

## Parámetros útiles

| Parámetro | Descripción |
|-----------|-------------|
| `cfg.imageSize` | Tamaño de entrenamiento (default `[512 512]`) |
| `cfg.forceTestPatientIds` | Pacientes reservados para prueba (ej. `{'20588680'}`) |
| `cfg.fullResTestPatientIds` | Evaluación extra a tamaño nativo tras ROI |
| `cfg.maxEpochs` | Épocas de entrenamiento |
| `cfg.denoiseMethod` | `'median'`, `'bilateral'` o `'none'` |
| `cfg.splitByPatient` | `true` evita fuga entre train/test |

## Autor

Bryan Andrade — Tesis de mejora de calidad en mamografías DICOM.
