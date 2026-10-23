"""
Fase 0 - Descarga de datos (Hackathon ISAC 2026).
Descarga eventos, freeze frames 360 y alineaciones de los partidos
de Andre Jardine dirigiendo al Club America, y arma la tabla de partidos.

Requisitos:
    pip install statsbombpy pandas pyarrow
    export SB_USERNAME=...   y   export SB_PASSWORD=...

Uso:
    python descargar_datos.py
Se puede volver a correr: se salta lo que ya existe y reintenta lo que fallo.
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

COMPETITION_ID = 73          # Liga MX
SEASON_IDS = [281, 317, 318]  # 2023/2024, 2024/2025, 2025/2026
TEAM = "Am[eé]rica"
COACH = "Jardine"

DATA = Path("data")
for sub in ["events", "frames", "lineups"]:
    (DATA / sub).mkdir(parents=True, exist_ok=True)


def guardar(df: pd.DataFrame, ruta_sin_ext: Path) -> None:
    """Guarda en parquet; si alguna columna anidada falla, usa pickle."""
    try:
        df.to_parquet(ruta_sin_ext.with_suffix(".parquet"), index=False)
    except Exception:
        df.to_pickle(ruta_sin_ext.with_suffix(".pkl"))


def existe(ruta_sin_ext: Path) -> bool:
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


# 1) Tabla de partidos de Jardine con el America ------------------------
print("Descargando calendario...")
todos = []
for sid in SEASON_IDS:
    m = con_reintentos(lambda: sb.matches(competition_id=COMPETITION_ID,
                                          season_id=sid, creds=CREDS))
    todos.append(m)
matches = pd.concat(todos, ignore_index=True)

def contiene(serie, texto):
    return serie.astype(str).str.contains(texto, case=False, na=False)

local = contiene(matches["home_team"], TEAM) & contiene(matches["home_managers"], COACH)
visita = contiene(matches["away_team"], TEAM) & contiene(matches["away_managers"], COACH)
jar = matches[local | visita].copy()

jar["america_local"] = contiene(jar["home_team"], TEAM)
jar["rival"] = jar["away_team"].where(jar["america_local"], jar["home_team"])
jar["goles_america"] = jar["home_score"].where(jar["america_local"], jar["away_score"])
jar["goles_rival"] = jar["away_score"].where(jar["america_local"], jar["home_score"])
jar["resultado"] = "E"
jar.loc[jar["goles_america"] > jar["goles_rival"], "resultado"] = "G"
jar.loc[jar["goles_america"] < jar["goles_rival"], "resultado"] = "P"

columnas = ["match_id", "match_date", "season", "competition_stage", "match_week",
            "home_team", "away_team", "home_managers", "away_managers",
            "america_local", "rival", "goles_america", "goles_rival", "resultado",
            "match_status_360"]
columnas = [c for c in columnas if c in jar.columns]
jar = jar[columnas].sort_values("match_date").reset_index(drop=True)
jar.to_csv(DATA / "partidos.csv", index=False)

print(f"Partidos de Jardine con el America: {len(jar)}")
print(jar.groupby("season").size().to_string())
print("Resultados:", jar["resultado"].value_counts().to_dict())

# 2) Eventos, 360 y alineaciones por partido ------------------------------
fallos = []
n = len(jar)
for i, row in jar.iterrows():
    mid = int(row["match_id"])
    etiqueta = f"[{i + 1}/{n}] {row['match_date']} vs {row['rival']} ({mid})"
    try:
        ruta = DATA / "events" / str(mid)
        if not existe(ruta):
            ev = con_reintentos(lambda: sb.events(match_id=mid, creds=CREDS))
            guardar(ev, ruta)

        ruta = DATA / "frames" / str(mid)
        if not existe(ruta) and row.get("match_status_360") == "available":
            fr = con_reintentos(lambda: sb.frames(match_id=mid, creds=CREDS))
            guardar(fr, ruta)

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

# 3) Resumen -------------------------------------------------------------
print("\nResumen:")
for sub in ["events", "frames", "lineups"]:
    print(f"  {sub}: {len(list((DATA / sub).glob('*')))} archivos")
if fallos:
    print(f"  Fallaron {len(fallos)} partidos: {fallos}. Vuelve a correr el script.")
else:
    print("  Sin errores.")
