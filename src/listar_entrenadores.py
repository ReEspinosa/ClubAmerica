"""
Lista los entrenadores de Liga MX con mas partidos dirigidos al mismo equipo
en los datos disponibles (para elegir casos de estudio o comparaciones).

Uso:
    python listar_entrenadores.py
Requiere SB_USERNAME y SB_PASSWORD en las variables de entorno.
"""
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

try:
    CREDS = {"user": os.environ["SB_USERNAME"], "passwd": os.environ["SB_PASSWORD"]}
except KeyError:
    sys.exit("Faltan SB_USERNAME / SB_PASSWORD en las variables de entorno.")

COMPETITION_ID = 73  # Liga MX varonil
comps = sb.competitions(creds=CREDS)
temporadas = comps[comps["competition_id"] == COMPETITION_ID]["season_id"].tolist()

partidos = []
for sid in temporadas:
    m = sb.matches(competition_id=COMPETITION_ID, season_id=sid, creds=CREDS)
    partidos.append(m)
matches = pd.concat(partidos, ignore_index=True)

# Una fila por (partido, equipo, entrenador)
filas = []
for lado in ["home", "away"]:
    tmp = matches[["match_id", "match_date", "season", "match_status_360",
                   f"{lado}_team", f"{lado}_managers"]].copy()
    tmp.columns = ["match_id", "match_date", "season", "status_360", "equipo", "entrenador"]
    filas.append(tmp)
largo = pd.concat(filas, ignore_index=True).dropna(subset=["entrenador"])

resumen = (largo.groupby(["entrenador", "equipo"])
           .agg(partidos=("match_id", "nunique"),
                con_360=("status_360", lambda s: (s == "available").sum()),
                desde=("match_date", "min"),
                hasta=("match_date", "max"),
                temporadas=("season", lambda s: ", ".join(sorted(s.unique()))))
           .reset_index()
           .sort_values("partidos", ascending=False))

Path("data").mkdir(exist_ok=True)
resumen.to_csv("data/entrenadores.csv", index=False)

print("Top 25 entrenadores por partidos dirigidos al mismo equipo:\n")
print(resumen.head(25).to_string(index=False))
print("\nGuardado: data/entrenadores.csv")
