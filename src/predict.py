import requests

# =====================================================
# FEATURES (MISMO ORDEN QUE EN TRAIN)
# =====================================================

FEATURES = [
    "ley_mot_alim_ro",
    "ley_cut_alim_ro",
    "ley_fe_alim_ro",
    "re_mo_total",
    "ley_cus_alim_ro"
]

# =====================================================
# INPUT DE PRUEBA
# =====================================================

data = [[
    800,   # ley_mot_alim_ro
    0.75,  # ley_cut_alim_ro
    5.0,   # ley_fe_alim_ro
    70,    # re_mo_total
    0.12   # ley_cus_alim_ro
]]

payload = {
    "dataframe_split": {
        "columns": FEATURES,
        "data": data
    }
}

# =====================================================
# REQUEST AL MODELO SERVIDO
# =====================================================

response = requests.post(
    "http://127.0.0.1:5001/invocations",
    json=payload
)

print("Respuesta del modelo:")
print(response.json())