"""
Contrafactual 360 (Hackathon ISAC 2026): en cada tiro, habia un companero
mejor colocado al que se le pudo pasar el balon?

Para cada tiro en juego abierto con foto 360:
  - Se calcula un xG aproximado para cada companero visible (modelo de distancia
    y angulo ajustado con el xG de StatsBomb de todos los tiros), castigado por
    los rivales que tapan su tiro.
  - El pase debe ser posible: a 30 yardas o menos, sin rivales a menos de 1.5
    yardas de la linea de pase y con el companero habilitado (no en fuera de juego).
  - Hay "mejor opcion" si ese xG supera al del tirador por 0.05 o mas.

Solo guarda resultados agregados.
Entradas (en data/): partidos.csv, events/, frames/
Salida:              data/contrafactual.json

Uso:  python contrafactual_tiros.py
"""
import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from construir_huella import col, leer, preparar

warnings.filterwarnings("ignore")
MARGEN = 0.05
PASE_MAX = 30.0
LINEA_PASE = 1.5


def geo(x, y):
    d = np.hypot(120 - x, 40 - y)
    a1 = np.arctan2(36 - y, 120 - x)
    a2 = np.arctan2(44 - y, 120 - x)
    ang = np.abs(a2 - a1)
    return d, ang


def x_modelo(X, coef):
    d, a = geo(X[:, 0], X[:, 1])
    z = coef[0] + coef[1] * d + coef[2] * a + coef[3] * np.log1p(d)
    return 1 / (1 + np.exp(-z))


def ajustar(tiros):
    """Regresion de logit(xG) sobre distancia y angulo (minimos cuadrados)."""
    xg = np.clip(tiros[:, 2], 0.005, 0.95)
    d, a = geo(tiros[:, 0], tiros[:, 1])
    A = np.column_stack([np.ones_like(d), d, a, np.log1p(d)])
    coef, *_ = np.linalg.lstsq(A, np.log(xg / (1 - xg)), rcond=None)
    return coef


def tapan(p, rivales):
    """Rivales dentro del triangulo jugador-postes."""
    if len(rivales) == 0:
        return 0
    n = 0
    for q in rivales:
        a, b, c = p, np.array([120, 36]), np.array([120, 44])
        v0, v1, v2 = c - a, b - a, q - a
        d00, d01, d02, d11, d12 = v0 @ v0, v0 @ v1, v0 @ v2, v1 @ v1, v1 @ v2
        den = d00 * d11 - d01 * d01
        if den == 0:
            continue
        u = (d11 * d02 - d01 * d12) / den
        v = (d00 * d12 - d01 * d02) / den
        n += (u >= 0) and (v >= 0) and (u + v <= 1)
    return n


def dist_segmento(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.hypot(*(a + t * ab - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    data = Path(args.data)
    partidos = pd.read_csv(data / "partidos.csv")

    # 1) Juntar tiros con foto 360
    registros, todos = [], []
    for rw in partidos.itertuples():
        mid, caso, equipo = int(rw.match_id), rw.caso, rw.equipo
        ev = leer(data / "events" / str(mid))
        fr = leer(data / "frames" / str(mid))
        if ev is None or ev.empty:
            continue
        ev = preparar(ev)
        sh = ev[(ev["type"] == "Shot") & ev["x"].notna() & (col(ev, "shot_type") == "Open Play")]
        for t in sh.itertuples():
            todos.append([t.x, t.y, float(getattr(t, "shot_statsbomb_xg", 0) or 0)])
        if fr is None or fr.empty:
            continue
        sh = sh[sh["team"] == equipo]
        f = fr[fr["id"].isin(sh["id"])]
        if f.empty:
            continue
        for sid, g in f.groupby("id"):
            loc = []
            for v in g["location"].values:
                try:
                    a = np.ravel(np.asarray(v, dtype=float))
                    loc.append(a[:2] if a.size >= 2 else [np.nan, np.nan])
                except (TypeError, ValueError):
                    loc.append([np.nan, np.nan])
            loc = np.array(loc)
            ok = np.isfinite(loc).all(1)
            tm, ac, kp = g["teammate"].values.astype(bool), g["actor"].values.astype(bool), g["keeper"].values.astype(bool)
            t = sh[sh["id"] == sid].iloc[0]
            registros.append({"caso": caso, "jugador": t["player"], "x": t["x"], "y": t["y"],
                              "xg": float(t.get("shot_statsbomb_xg", 0) or 0),
                              "gol": t.get("shot_outcome") == "Goal",
                              "comp": loc[ok & tm & ~ac], "riv": loc[ok & ~tm]})
    todos = np.array(todos)
    coef = ajustar(todos[todos[:, 2] > 0])
    print(f"Tiros en juego abierto para el modelo: {len(todos)}; con foto 360 del equipo analizado: {len(registros)}")

    # 2) Evaluar opciones
    res = defaultdict(lambda: {"tiros": 0, "con_opcion": 0, "xg_tirado": 0.0, "xg_opcion": 0.0, "goles_con_opcion": 0})
    jug = defaultdict(lambda: {"tiros": 0, "con_opcion": 0, "ganancia": 0.0})
    zonas = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in registros:
        tirador = np.array([r["x"], r["y"]])
        riv = r["riv"]
        ult = np.sort(riv[:, 0])[-2] if len(riv) >= 2 else 120
        mejor = 0.0
        for p in r["comp"]:
            if np.hypot(*(p - tirador)) > PASE_MAX or p[0] > max(ult, tirador[0]):
                continue
            if len(riv) and min(dist_segmento(q, tirador, p) for q in riv) < LINEA_PASE:
                continue
            xg_p = float(x_modelo(p[None, :], coef)[0]) * (0.65 ** tapan(p, riv))
            mejor = max(mejor, xg_p)
        hay = mejor >= r["xg"] + MARGEN
        for c in (r["caso"],):
            o = res[c]
            o["tiros"] += 1; o["xg_tirado"] += r["xg"]
            if hay:
                o["con_opcion"] += 1; o["xg_opcion"] += mejor - r["xg"]; o["goles_con_opcion"] += int(r["gol"])
        k = (r["caso"], r["jugador"])
        jug[k]["tiros"] += 1
        if hay:
            jug[k]["con_opcion"] += 1; jug[k]["ganancia"] += mejor - r["xg"]
        zona = "Area chica" if r["x"] >= 114 else ("Area" if r["x"] >= 102 and 18 <= r["y"] <= 62 else "Fuera del area")
        zonas[r["caso"]][zona][0] += 1; zonas[r["caso"]][zona][1] += int(hay)

    salida = {"casos": {}, "jugadores": {}, "parametros": {"margen": MARGEN, "pase_max": PASE_MAX, "linea_pase": LINEA_PASE}}
    print("\nCaso                    tiros  con mejor opcion  xG extra por tiro con opcion  goles en esos tiros")
    for c, o in res.items():
        pct = o["con_opcion"] / max(o["tiros"], 1)
        salida["casos"][c] = {"tiros": o["tiros"], "pct_con_opcion": round(pct, 4),
                              "xg_extra_medio": round(o["xg_opcion"] / max(o["con_opcion"], 1), 4),
                              "xg_extra_total_por_tiro": round(o["xg_opcion"] / max(o["tiros"], 1), 4),
                              "goles_con_opcion": o["goles_con_opcion"],
                              "zonas": {z: {"tiros": v[0], "pct": round(v[1] / max(v[0], 1), 4)} for z, v in zonas[c].items()}}
        print(f"{c:<22} {o['tiros']:>6}  {pct:>15.1%}  {salida['casos'][c]['xg_extra_medio']:>27.3f}  {o['goles_con_opcion']:>8}")
        for z, v in zonas[c].items():
            print(f"      {z:<16} {v[0]:>5} tiros   {v[1] / max(v[0], 1):.1%} con mejor opcion")
    for c in res:
        js = [(j, v) for (cc, j), v in jug.items() if cc == c and v["tiros"] >= 15]
        js.sort(key=lambda x: -x[1]["con_opcion"] / x[1]["tiros"])
        salida["jugadores"][c] = [{"jugador": j, "tiros": v["tiros"], "pct": round(v["con_opcion"] / v["tiros"], 3),
                                   "xg_extra": round(v["ganancia"], 2)} for j, v in js]
        if js:
            print(f"\n  {c}: jugadores (15+ tiros) que mas tiraron teniendo mejor opcion")
            for j, v in js[:5]:
                print(f"    {j:<40} {v['tiros']:>4} tiros  {v['con_opcion'] / v['tiros']:.0%}  (+{v['ganancia']:.2f} xG)")
    with open(data / "contrafactual.json", "w", encoding="utf-8") as fh:
        json.dump(salida, fh, ensure_ascii=False)
    print(f"\nGuardado: {data / 'contrafactual.json'}")


if __name__ == "__main__":
    main()
