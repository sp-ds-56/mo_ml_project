# Proyecto MLflow - Predicción Ley Concentrado Mo

## Objetivo

Predecir la variable:

ley_conc_mo_total_4ta_limpieza

mediante un modelo XGBoost.

## Variables de entrada

- ley_mot_alim_ro
- ley_cut_alim_ro
- ley_fe_alim_ro
- re_mo_total
- ley_cus_alim_ro

## Tecnologías

- Python
- XGBoost
- MLflow
- Scikit-Learn

## Entrenamiento

```bash
python src/train.py
```

## Tracking de experimentos

```bash
mlflow ui
```

Abrir:

```text
http://localhost:5000
```

## Predicción online

```bash
python src/predict.py
```

## Predicción batch

```bash
python src/batch_predict.py
```