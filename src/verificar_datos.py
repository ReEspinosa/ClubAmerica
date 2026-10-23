"""
Verificación de acceso a la API de StatsBomb (Hackathon ISAC 2026).
Responde: ¿entran las credenciales?, ¿qué competiciones/temporadas hay?,
¿cuántos partidos de Jardine con el América?, ¿cuántos tienen 360?
Las credenciales se leen de variables de entorno (NUNCA en el código).
"""
import os
import sys
import warnings

import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

try:
    CREDS = {"user": os.environ["SB_USERNAME"], "passwd": os.environ["SB_PASSWORD"]}
except KeyError:
    sys.exit("Faltan SB_USERNAME / SB_PASSWORD en las variables de entorno.")

# 1) Competiciones disponibles -------------------------------------------
print("\n=== 1. Competiciones y temporadas disponibles ===")
comps = sb.competitions(creds=CREDS)
cols = [c for c in ["competition_id", "season_id", "country_name", "competition_name",
                    "season_name", "match_available_360"] if c in comps.columns]
print(comps[cols].to_string(index=False))

# 2) Partidos de todas las temporadas disponibles ------------------------
print("\n=== 2. Descargando lista de partidos por temporada ===")
partidos = []
for _, r in comps.iterrows():
    try:
        m = sb.matches(competition_id=r.competition_id, season_id=r.season_id, creds=CREDS)
        m["competition_name"] = r.competition_name
        m["season_name"] = r.season_name
        partidos.append(m)
        print(f"  {r.competition_name} {r.season_name}: {len(m)} partidos")
    except Exception as e:
        print(f"  {r.competition_name} {r.season_name}: error -> {e}")

matches = pd.concat(partidos, ignore_index=True)

# 3) Filtrar América y Jardine -------------------------------------------
es_america = (matches["home_team"].str.contains("América|America", case=False, na=False)
              | matches["away_team"].str.contains("América|America", case=False, na=False))
america = matches[es_america].copy()

mgr_cols = [c for c in ["home_managers", "away_managers"] if c in america.columns]
if mgr_cols:
    managers = america[mgr_cols].astype(str).agg(" | ".join, axis=1)
    jardine = america[managers.str.contains("Jardine", case=False, na=False)].copy()
else:
    print("\n(No vienen columnas de entrenador; se muestran todos los partidos del América)")
    jardine = america

print(f"\n=== 3. Partidos del América: {len(america)} | con Jardine: {len(jardine)} ===")
if len(jardine):
    print(jardine.groupby(["competition_name", "season_name"]).size().to_string())
    if "match_status_360" in jardine.columns:
        print("\nEstado de datos 360:")
        print(jardine["match_status_360"].value_counts().to_string())
    print("\nRango de fechas:", jardine["match_date"].min(), "->", jardine["match_date"].max())
    jardine.sort_values("match_date").to_csv("partidos_jardine.csv", index=False)
    print("Guardado: partidos_jardine.csv")

# 4) Probar eventos y 360 de un partido ----------------------------------
if len(jardine):
    mid = int(jardine.sort_values("match_date").iloc[-1]["match_id"])
    print(f"\n=== 4. Prueba con el partido {mid} ===")
    ev = sb.events(match_id=mid, creds=CREDS)
    print(f"Eventos: {len(ev)} filas, {ev.shape[1]} columnas")
    print("Tipos más comunes:\n", ev["type"].value_counts().head(10).to_string())
    obv = [c for c in ev.columns if "obv" in c]
    print("Columnas OBV:", obv if obv else "no vienen")
    try:
        fr = sb.frames(match_id=mid, creds=CREDS)
        print(f"Freeze frames 360: {len(fr)} filas -> 360 OK")
    except Exception as e:
        print("360 no disponible para este partido:", e)

print("\nListo ✅")
