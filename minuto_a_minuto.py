"""
Minuto a minuto (Hackathon ISAC 2026).

Calcula, para cada partido de los casos, el "momentum" minuto a minuto:
    - xT generado por cada equipo en cada minuto (cadena de Markov de construir_huella.py)
    - tiros con su xG y minuto
    - goles, cambios y expulsiones
Y, por caso, el "partido tipico": dominio y xG por bloques de 15 minutos.

Solo guarda resultados agregados (no datos crudos de StatsBomb).

Entradas:  data/partidos.csv, data/events/*, data/xt_grid.npy
Salida:    data/minuto_a_minuto.json

Uso (en la misma carpeta que construir_huella.py):
    python minuto_a_minuto.py
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from construir_huella import XT_NX, XT_NY, celda, col, es_juego_abierto, leer, preparar

warnings.filterwarnings("ignore")

FIN_PERIODO = {1: 45, 2: 90, 3: 105, 4: 120}
BLOQUES = ["0-15", "15-30", "30-45+", "45-60", "60-75", "75-90+"]


def reloj(minuto, periodo):
    """Minuto como se ve en el marcador: 23', 45+2', 90+4'."""
    fin = FIN_PERIODO.get(int(periodo), 90)
    m = int(minuto)
    return f"{fin}+{m - fin + 1}'" if m >= fin else f"{m + 1}'"


def bloque(minuto, periodo):
    m, p = int(minuto), int(periodo)
    if p == 1:
        return 0 if m < 15 else 1 if m < 30 else 2
    if p == 2:
        return 3 if m < 60 else 4 if m < 75 else 5
    return None  # tiempos extra fuera de los bloques


def xt_ganado(ev, xt):
    """xT positivo ganado por pases completos en juego abierto y conducciones."""
    pa = ev[(ev["type"] == "Pass") & col(ev, "pass_outcome").isna() & es_juego_abierto(ev)
            & ev["x"].notna() & ev["pex"].notna()]
    ca = ev[(ev["type"] == "Carry") & ev["x"].notna() & ev["cex"].notna()]
    movs = pd.concat([pa.assign(ex=pa["pex"], ey=pa["pey"]), ca.assign(ex=ca["cex"], ey=ca["cey"])])
    if movs.empty:
        return movs.assign(dxt=[])
    ix, iy = celda(movs["x"].values, movs["y"].values, XT_NX, XT_NY)
    jx, jy = celda(movs["ex"].values, movs["ey"].values, XT_NX, XT_NY)
    movs["dxt"] = np.clip(xt[jx, jy] - xt[ix, iy], 0, None)
    return movs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    data = Path(args.data)

    partidos = pd.read_csv(data / "partidos.csv")
    xt = np.load(data / "xt_grid.npy")
    salida = {"partidos": [], "bloques": {}}
    acum = {}  # caso -> lista de (bloque, xt_propio, xt_rival, xg_propio, xg_rival)

    filas = partidos.sort_values("match_date").itertuples()
    for k, r in enumerate(filas, 1):
        ev = leer(data / "events" / str(int(r.match_id)))
        if ev is None or ev.empty:
            continue
        ev = preparar(ev)
        ev["min_abs"] = (ev["t_abs"] // 60).astype(int)
        n_min = int(ev["min_abs"].max()) + 1
        equipo, rival = r.equipo, r.rival

        movs = xt_ganado(ev, xt)
        serie = {}
        for eq in (equipo, rival):
            s = movs[movs["team"] == eq].groupby("min_abs")["dxt"].sum()
            serie[eq] = [round(float(s.get(m, 0.0)), 4) for m in range(n_min)]

        tiros = ev[(ev["type"] == "Shot")]
        lista_tiros = [{
            "m": int(t.min_abs), "reloj": reloj(t.minute, t.period),
            "eq": 0 if t.team == equipo else 1,
            "xg": round(float(col(tiros, "shot_statsbomb_xg").get(t.Index, 0) or 0), 3),
            "gol": bool(col(tiros, "shot_outcome").get(t.Index) == "Goal"),
            "pen": bool(col(tiros, "shot_type").get(t.Index) == "Penalty"),
            "jug": str(col(tiros, "player").get(t.Index, "")),
        } for t in tiros.itertuples()]

        marcas = []
        for t in ev[ev["type"] == "Own Goal For"].itertuples():
            marcas.append({"m": int(t.min_abs), "reloj": reloj(t.minute, t.period),
                           "eq": 0 if t.team == equipo else 1, "tipo": "autogol"})
        for t in ev[ev["type"] == "Substitution"].itertuples():
            entra = col(ev, "substitution_replacement").get(t.Index, "")
            marcas.append({"m": int(t.min_abs), "reloj": reloj(t.minute, t.period),
                           "eq": 0 if t.team == equipo else 1, "tipo": "cambio",
                           "txt": f"{entra} por {t.player}"})
        tarjetas = pd.concat([col(ev, "foul_committed_card"), col(ev, "bad_behaviour_card")], axis=1)
        roja = tarjetas.apply(lambda f: any(isinstance(v, str) and ("Red" in v) for v in f), axis=1)
        for t in ev[roja].itertuples():
            marcas.append({"m": int(t.min_abs), "reloj": reloj(t.minute, t.period),
                           "eq": 0 if t.team == equipo else 1, "tipo": "roja", "txt": str(t.player)})

        cortes = [int(g["min_abs"].min()) for _, g in ev.groupby("period")][1:]
        salida["partidos"].append({
            "id": int(r.match_id), "caso": r.caso, "fecha": str(r.match_date),
            "equipo": equipo, "rival": rival, "local": bool(r.es_local),
            "goles": [int(r.goles_equipo), int(r.goles_rival)],
            "xt": [serie[equipo], serie[rival]], "tiros": lista_tiros,
            "marcas": marcas, "cortes_periodo": cortes,
        })

        # Acumular por bloques de 15 minutos para el partido tipico
        movs["bloque"] = [bloque(m, p) for m, p in zip(movs["minute"], movs["period"])]
        tiros_np = tiros[col(tiros, "shot_type") != "Penalty"].copy()
        tiros_np["bloque"] = [bloque(m, p) for m, p in zip(tiros_np["minute"], tiros_np["period"])]
        for b in range(6):
            xa = movs[(movs["bloque"] == b) & (movs["team"] == equipo)]["dxt"].sum()
            xb = movs[(movs["bloque"] == b) & (movs["team"] == rival)]["dxt"].sum()
            ga = col(tiros_np[(tiros_np["bloque"] == b) & (tiros_np["team"] == equipo)], "shot_statsbomb_xg").sum()
            gb = col(tiros_np[(tiros_np["bloque"] == b) & (tiros_np["team"] == rival)], "shot_statsbomb_xg").sum()
            acum.setdefault(r.caso, []).append((b, xa, xb, ga, gb))
        if k % 50 == 0:
            print(f"  {k} partidos...")

    for caso, filas_c in acum.items():
        df = pd.DataFrame(filas_c, columns=["b", "xa", "xb", "ga", "gb"])
        df["dom"] = df["xa"] / (df["xa"] + df["xb"]).replace(0, np.nan)
        res = []
        for b, g in df.groupby("b"):
            dom = g["dom"].dropna().values
            boots = [np.random.choice(dom, len(dom)).mean() for _ in range(1000)] if len(dom) > 2 else [np.nan]
            res.append({"bloque": BLOQUES[b], "dominio": round(float(dom.mean()), 4),
                        "ic_bajo": round(float(np.percentile(boots, 5)), 4),
                        "ic_alto": round(float(np.percentile(boots, 95)), 4),
                        "xg_favor": round(float(g["ga"].mean()), 4), "xg_contra": round(float(g["gb"].mean()), 4),
                        "partidos": int(len(g))})
        salida["bloques"][caso] = res

    with open(data / "minuto_a_minuto.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, separators=(",", ":"))
    tam = (data / "minuto_a_minuto.json").stat().st_size / 1e6
    print(f"\nGuardado: {data / 'minuto_a_minuto.json'} ({len(salida['partidos'])} filas, {tam:.1f} MB)")
    print("\nDominio por bloque de 15 minutos (proporcion del xT del partido):")
    tabla = {c: {x["bloque"]: x["dominio"] for x in v} for c, v in salida["bloques"].items()}
    print(pd.DataFrame(tabla).T[BLOQUES].round(3).to_string())


if __name__ == "__main__":
    main()
