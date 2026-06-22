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

## Parámetros útiles

| Parámetro | Descripción |
|-----------|-------------|
| `cfg.imageSize` | Tamaño de entrada (bajar a `[256 256]` si hay poca RAM) |
| `cfg.maxEpochs` | Épocas de entrenamiento |
| `cfg.denoiseMethod` | `'median'`, `'bilateral'` o `'none'` |
| `cfg.splitByPatient` | `true` evita fuga entre train/test |

## Autor

Bryan Andrade — Tesis de mejora de calidad en mamografías DICOM.
