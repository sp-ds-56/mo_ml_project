import pandas as pd
import joblib

# =====================================================
# MODELO
# =====================================================

model = joblib.load("models/model.pkl")

# =====================================================
# FEATURES (OBLIGATORIO IGUAL QUE TRAIN)
# =====================================================

FEATURES = [
    "ley_mot_alim_ro",
    "ley_cut_alim_ro",
    "ley_fe_alim_ro",
    "re_mo_total",
    "ley_cus_alim_ro"
]

# =====================================================
# DATA
# =====================================================

df = pd.read_csv("data/new_data.csv")

# solo features
X = df[FEATURES].copy()

# =====================================================
# PREDICCIÓN
# =====================================================

pred = model.predict(X)

df["prediction"] = pred

# =====================================================
# GUARDAR RESULTADO
# =====================================================

df.to_pickle("data/predictions.pkl")

print("Predicciones generadas.")