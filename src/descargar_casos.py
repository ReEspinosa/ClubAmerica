"""
Fase 0 (ampliada) - Descarga de todos los casos de estudio (Hackathon ISAC 2026).

Casos:
    Jardine-America, Jardine-SanLuis,
    Almada-Pachuca, Almada-Santos, Almada-America,
    Ortiz-America (como era el America antes de Jardine)

Descarga eventos, freeze frames 360 y alineaciones de cada partido.
Cada archivo de eventos trae a ambos equipos, asi que los rivales sirven
como referencia de "la liga" sin descargar nada extra.

Uso:
    python descargar_casos.py
Se puede volver a correr: se salta lo que ya existe y reintenta lo que fallo.
Requiere SB_USERNAME y SB_PASSWORD en las variables de entorno.
"""
import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore")

try:
    CREDS = {"user": os.environ["SB_USERNAME"], "passwd": os.environ["SB_PASSWORD"]}
except KeyError:
    sys.exit("Faltan SB_USERNAME / SB_PASSWORD en las variables de entorno.")

COMPETITION_ID = 73  # Liga MX varonil

# (nombre del caso, apellido del entrenador, nombre exacto del equipo en StatsBomb)
CASOS = [
    ("Jardine-America", "Jardine", "América"),
    ("Jardine-SanLuis", "Jardine", "Atlético San Luis"),
    ("Almada-Pachuca", "Almada", "Pachuca"),
    ("Almada-Santos", "Almada", "Santos Laguna"),
    ("Almada-America", "Almada", "América"),
    ("Ortiz-America", "Fernando Ortiz", "América"),
]

DATA = Path("data")
for sub in ["events", "frames", "lineups"]:
    (DATA / sub).mkdir(parents=True, exist_ok=True)


def guardar(df, ruta_sin_ext):
    try:
        df.to_parquet(ruta_sin_ext.with_suffix(".parquet"), index=False)
    except Exception:
        df.to_pickle(ruta_sin_ext.with_suffix(".pkl"))


def existe(ruta_sin_ext):
    return (ruta_sin_ext.with_suffix(".parquet").exists()
            or ruta_sin_ext.with_suffix(".pkl").exists())


def con_reintentos(fn, intentos=3, espera=5):
    for i in range(intentos):
        try:
            return fn()
        except Exception as e:
            if i == intentos - 1:
                raise
            print(f"    reintento {i + 1} tras error: {e}")
            time.sleep(espera)


# 1) Calendario completo de Liga MX varonil -------------------------------
print("Descargando calendario de todas las temporadas...")
comps = sb.competitions(creds=CREDS)
temporadas = comps[comps["competition_id"] == COMPETITION_ID]["season_id"].tolist()
todos = []
for sid in temporadas:
    todos.append(con_reintentos(
        lambda: sb.matches(competition_id=COMPETITION_ID, season_id=sid, creds=CREDS)))
matches = pd.concat(todos, ignore_index=True)

# 2) Tabla de partidos por caso -------------------------------------------
filas = []
for caso, entrenador, equipo in CASOS:
    for lado, otro in [("home", "away"), ("away", "home")]:
        sel = matches[
            (matches[f"{lado}_team"] == equipo)
            & matches[f"{lado}_managers"].astype(str).str.contains(entrenador, case=False, na=False)
        ].copy()
        if sel.empty:
            continue
        sel["caso"] = caso
        sel["entrenador"] = entrenador
        sel["equipo"] = equipo
        sel["es_local"] = lado == "home"
        sel["rival"] = sel[f"{otro}_team"]
        sel["goles_equipo"] = sel[f"{lado}_score"]
        sel["goles_rival"] = sel[f"{otro}_score"]
        filas.append(sel)

partidos = pd.concat(filas, ignore_index=True)
partidos["resultado"] = "E"
partidos.loc[partidos["goles_equipo"] > partidos["goles_rival"], "resultado"] = "G"
partidos.loc[partidos["goles_equipo"] < partidos["goles_rival"], "resultado"] = "P"

columnas = ["caso", "entrenador", "equipo", "match_id", "match_date", "season",
            "competition_stage", "match_week", "es_local", "rival",
            "goles_equipo", "goles_rival", "resultado", "match_status_360"]
columnas = [c for c in columnas if c in partidos.columns]
partidos = partidos[columnas].sort_values(["caso", "match_date"]).reset_index(drop=True)
partidos.to_csv(DATA / "partidos.csv", index=False)

print("\nPartidos por caso:")
print(partidos.groupby("caso").agg(
    partidos=("match_id", "nunique"),
    con_360=("match_status_360", lambda s: (s == "available").sum()),
    desde=("match_date", "min"),
    hasta=("match_date", "max")).to_string())

# 3) Descarga por partido (sin repetir partidos compartidos entre casos) ----
unicos = partidos.drop_duplicates("match_id").reset_index(drop=True)
fallos = []
n = len(unicos)
for i, row in unicos.iterrows():
    mid = int(row["match_id"])
    etiqueta = f"[{i + 1}/{n}] {row['caso']} {row['match_date']} vs {row['rival']} ({mid})"
    try:
        ruta = DATA / "events" / str(mid)
        if not existe(ruta):
            guardar(con_reintentos(lambda: sb.events(match_id=mid, creds=CREDS)), ruta)

        ruta = DATA / "frames" / str(mid)
        if not existe(ruta) and row.get("match_status_360") == "available":
            guardar(con_reintentos(lambda: sb.frames(match_id=mid, creds=CREDS)), ruta)

        ruta = DATA / "lineups" / str(mid)
        if not existe(ruta):
            lu = con_reintentos(lambda: sb.lineups(match_id=mid, creds=CREDS))
            lu = pd.concat([df.assign(team=team) for team, df in lu.items()],
                           ignore_index=True)
            guardar(lu, ruta)

        print(f"{etiqueta} OK")
    except Exception as e:
        print(f"{etiqueta} ERROR: {e}")
        fallos.append(mid)

# 4) Resumen ---------------------------------------------------------------
print("\nResumen:")
for sub in ["events", "frames", "lineups"]:
    print(f"  {sub}: {len(list((DATA / sub).glob('*')))} archivos")
if fallos:
    print(f"  Fallaron {len(fallos)} partidos: {fallos}. Vuelve a correr el script.")
else:
    print("  Sin errores.")
