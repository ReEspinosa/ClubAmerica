"""
Partido tipico y puntos debiles (Hackathon ISAC 2026).

1) Partido tipico por tramos de 15 minutos, para cada caso:
   - posicion media de cada rol con balon (acciones con balon) y sin balon (acciones defensivas)
   - pases entre roles (para simular jugadas sobre la red de pases real de ese tramo)
   - mapa de calor de acciones con balon y de presiones
   - tiros a favor y en contra (ubicacion y xG)
2) Puntos debiles (como atacarlo), para cada caso:
   - por que carril le entran al ultimo tercio y por cual le generan mas xG
   - desde donde le rematan
   - donde pierde el balon y cuanto le cuesta cada perdida
   - de que tipo de jugada vienen los goles y el xG que concede

Entradas: data/partidos.csv, data/events/   Salida: data/partido_tipico.json (solo agregados)
Uso (misma carpeta que construir_huella.py y analisis_completo.py):
    python partido_tipico.py
"""
import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from construir_huella import col, es_juego_abierto, leer, preparar
from analisis_completo import ROL, NOMBRE_ROL, JUEGO_ABIERTO, VENTANA_TRANS, etapa, posesiones

warnings.filterwarnings("ignore")
BLOQUES = ["0-15", "15-30", "30-45+", "45-60", "60-75", "75-90+"]
GX, GY = 12, 8
CON_BALON = {"Pass", "Carry", "Ball Receipt*", "Shot", "Dribble"}
SIN_BALON = {"Pressure", "Duel", "Interception", "Ball Recovery", "Block", "Clearance", "Foul Committed"}
ORIGEN = {"Regular Play": "Juego elaborado", "From Counter": "Contragolpe", "From Corner": "Corner",
          "From Free Kick": "Tiro libre", "From Throw In": "Saque de banda", "From Goal Kick": "Saque de meta",
          "From Keeper": "Saque del portero", "From Kick Off": "Saque de centro", "Other": "Otro (incluye penales)"}


def bloque(minuto, periodo):
    m, p = int(minuto), int(periodo)
    if p == 1:
        return 0 if m < 15 else 1 if m < 30 else 2
    if p == 2:
        return 3 if m < 60 else 4 if m < 75 else 5
    return None


def carril(y):
    """Carril visto desde el equipo que ataca (y=0 es su izquierda)."""
    return "izq" if y < 80 / 3 else ("centro" if y <= 160 / 3 else "der")


def r(v, d=3):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    data = Path(args.data)
    partidos = pd.read_csv(data / "partidos.csv")

    on = defaultdict(lambda: np.zeros(3))      # (g, b, rol) -> [sx, sy, n]
    off = defaultdict(lambda: np.zeros(3))
    aristas = defaultdict(float)               # (g, b, a, c) -> pases
    heat_on = defaultdict(lambda: np.zeros((GX, GY)))
    heat_pr = defaultdict(lambda: np.zeros((GX, GY)))
    tiros_f = defaultdict(list)
    tiros_c = defaultdict(list)
    n_part = defaultdict(int)
    # debilidades
    ent = defaultdict(lambda: defaultdict(float))      # entradas del rival al ultimo tercio por carril
    xg_carril = defaultdict(lambda: defaultdict(float))
    perd = defaultdict(lambda: np.zeros((6, 4, 3)))    # zona de perdida -> [n, tiros rival 10s, xG]
    origen = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))  # tipo -> [xG, goles, tiros]
    tiros_rival_todos = defaultdict(list)
    minutos = defaultdict(float)

    filas = list(partidos.sort_values("match_date").itertuples())
    print(f"Procesando {len(filas)} partidos...")
    for k, rw in enumerate(filas, 1):
        ev = leer(data / "events" / str(int(rw.match_id)))
        if ev is None or ev.empty:
            continue
        ev = preparar(ev)
        equipo, caso = rw.equipo, rw.caso
        et = etapa(caso, rw.match_date)
        grupos = [caso] + ([f"{caso}|{et}"] if et else [])
        ev["bl"] = [bloque(m, p) if not np.isnan(m) else None for m, p in zip(ev["minute"], ev["period"])]
        q, rv = ev[ev["team"] == equipo], ev[ev["team"] != equipo]
        rol_j = q.dropna(subset=["position"]).groupby("player")["position"].agg(lambda s: s.mode().iloc[0]).map(ROL)
        q = q.assign(rol=q["player"].map(rol_j))
        for g in grupos:
            n_part[g] += 1
            minutos[g] += ev.attrs["duracion"] / 60

        ok = q["x"].notna() & q["bl"].notna() & q["rol"].notna()
        a_on = q[ok & q["type"].isin(CON_BALON)]
        a_off = q[ok & q["type"].isin(SIN_BALON)]
        for dest, df in ((on, a_on), (off, a_off)):
            agg = df.groupby(["bl", "rol"]).agg(sx=("x", "sum"), sy=("y", "sum"), n=("x", "size"))
            for (b, rl), v in agg.iterrows():
                for g in grupos:
                    dest[(g, int(b), rl)] += [v.sx, v.sy, v.n]
        pa = q[(q["type"] == "Pass") & col(q, "pass_outcome").isna() & es_juego_abierto(q) & q["bl"].notna()]
        pa = pa.assign(r_a=col(pa, "pass_recipient").map(rol_j)).dropna(subset=["rol", "r_a"])
        pa = pa[pa["rol"] != pa["r_a"]]
        for (b, a, c), n in pa.groupby(["bl", "rol", "r_a"]).size().items():
            for g in grupos:
                aristas[(g, int(b), a, c)] += n
        for dest, df in ((heat_on, a_on), (heat_pr, q[ok & (q["type"] == "Pressure")])):
            ix = np.clip((df["x"] / 120 * GX).astype(int), 0, GX - 1)
            iy = np.clip((df["y"] / 80 * GY).astype(int), 0, GY - 1)
            for b in range(6):
                m = (df["bl"] == b).values
                h = np.zeros((GX, GY))
                np.add.at(h, (ix[m], iy[m]), 1)
                for g in grupos:
                    dest[(g, b)] += h
        sh = ev[(ev["type"] == "Shot") & ev["bl"].notna() & ev["x"].notna()]
        for t in sh.itertuples():
            fila = [round(t.x, 1), round(t.y, 1), round(float(getattr(t, "shot_statsbomb_xg", 0) or 0), 3),
                    int(getattr(t, "shot_outcome", "") == "Goal")]
            for g in grupos:
                (tiros_f if t.team == equipo else tiros_c)[(g, int(t.bl))].append(fila)

        # ---- debilidades ----
        rv_mov = rv[(rv["type"].isin(["Pass", "Carry"])) & rv["x"].notna()]
        ex = np.where(rv_mov["type"] == "Pass", rv_mov["pex"], rv_mov["cex"])
        ey = np.where(rv_mov["type"] == "Pass", rv_mov["pey"], rv_mov["cey"])
        okm = (rv_mov["type"] == "Carry") | col(rv_mov, "pass_outcome").isna()
        entra = okm.values & (rv_mov["x"].values < 80) & (ex >= 80)
        for yy in ey[entra]:
            if not np.isnan(yy):
                for g in grupos:
                    ent[g][carril(yy)] += 1
        rsh = rv[(rv["type"] == "Shot") & rv["x"].notna()]
        for t in rsh.itertuples():
            xg = float(getattr(t, "shot_statsbomb_xg", 0) or 0)
            for g in grupos:
                tiros_rival_todos[g].append([round(t.x, 1), round(t.y, 1), round(xg, 3), int(getattr(t, "shot_outcome", "") == "Goal"),
                                             ORIGEN.get(t.play_pattern, "Otro (incluye penales)")])
                if getattr(t, "shot_type", "") != "Penalty":
                    xg_carril[g][carril(t.y)] += xg
        p = posesiones(ev)
        tiros = ev[ev["type"] == "Shot"][["team", "t_abs", "possession"]].assign(
            xg=col(ev[ev["type"] == "Shot"], "shot_statsbomb_xg").fillna(0).values)
        ult = ev.groupby("possession").tail(1).set_index("possession")
        for i in range(len(p) - 1):
            a, b = p.iloc[i], p.iloc[i + 1]
            if a["equipo"] != equipo or b["equipo"] == equipo or b["patron"] not in JUEGO_ABIERTO:
                continue
            le = ult.loc[a["possession"]] if a["possession"] in ult.index else None
            if le is None or np.isnan(le["x"]):
                continue
            x0, y0 = (le["x"], le["y"]) if le["team"] == equipo else (120 - le["x"], 80 - le["y"])
            zx, zy = min(5, int(x0 / 20)), min(3, int(y0 / 20))
            ts = tiros[(tiros["possession"] == b["possession"]) & (tiros["t_abs"] - b["t0"] <= VENTANA_TRANS)]
            for g in grupos:
                perd[g][zx, zy] += [1, len(ts), ts["xg"].sum()]
        for t in rsh.itertuples():
            o = ORIGEN.get(t.play_pattern, "Otro (incluye penales)")
            for g in grupos:
                origen[g][o] += [float(getattr(t, "shot_statsbomb_xg", 0) or 0), int(getattr(t, "shot_outcome", "") == "Goal"), 1]
        if k % 50 == 0:
            print(f"  {k}/{len(filas)}")

    out = {"bloques": BLOQUES, "tipico": {}, "debilidades": {}}
    for g in sorted(n_part):
        n = n_part[g]
        tramos = []
        for b in range(6):
            nodos_on = [{"rol": rl, "x": r(v[0] / v[2], 1), "y": r(v[1] / v[2], 1), "n": r(v[2] / n, 1)}
                        for (gg, bb, rl), v in on.items() if gg == g and bb == b and v[2] / n >= 1.0]
            nodos_off = [{"rol": rl, "x": r(v[0] / v[2], 1), "y": r(v[1] / v[2], 1), "n": r(v[2] / n, 2)}
                         for (gg, bb, rl), v in off.items() if gg == g and bb == b and v[2] / n >= 0.2]
            ar = [[a, c, r(w / n, 2)] for (gg, bb, a, c), w in aristas.items() if gg == g and bb == b and w / n >= 0.3]
            ho, hp = heat_on[(g, b)], heat_pr[(g, b)]
            tramos.append({
                "con_balon": sorted(nodos_on, key=lambda z: -z["n"])[:11],
                "sin_balon": sorted(nodos_off, key=lambda z: -z["n"])[:11],
                "pases": sorted(ar, key=lambda z: -z[2])[:40],
                "calor": (np.round(ho / max(ho.max(), 1), 3)).T.tolist(),
                "presion": (np.round(hp / max(hp.max(), 1), 3)).T.tolist(),
                "presiones_partido": r(hp.sum() / n, 1),
                "acciones_partido": r(ho.sum() / n, 1),
                "tiros_favor": tiros_f[(g, b)],
                "tiros_contra": tiros_c[(g, b)],
            })
        out["tipico"][g] = {"partidos": n, "tramos": tramos}

        tot_ent = sum(ent[g].values()) or 1
        pz = perd[g]
        zonas_perdida = []
        for zx in range(6):
            for zy in range(4):
                nn, ts, xg = pz[zx, zy]
                if nn:
                    zonas_perdida.append({"zx": zx, "zy": zy, "perdidas90": r(nn / minutos[g] * 90, 2),
                                          "pct_tiro": r(ts / nn, 3), "xg_por_perdida": r(xg / nn, 4)})
        out["debilidades"][g] = {
            "partidos": n,
            "entradas_carril": {c: r(v / tot_ent, 3) for c, v in ent[g].items()},
            "entradas_partido": r(tot_ent / n, 1),
            "xg_carril_partido": {c: r(v / n, 3) for c, v in xg_carril[g].items()},
            "tiros_rival": tiros_rival_todos[g],
            "perdidas": zonas_perdida,
            "origen": {o: {"xg_partido": r(v[0] / n, 3), "goles": int(v[1]), "tiros_partido": r(v[2] / n, 2)}
                       for o, v in sorted(origen[g].items(), key=lambda kv: -kv[1][0])},
        }

    with open(data / "partido_tipico.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\nGuardado: {data / 'partido_tipico.json'} ({(data / 'partido_tipico.json').stat().st_size / 1e6:.2f} MB)")
    print("\nPor donde le entran al ultimo tercio (parte de las entradas del rival; izq/der desde el rival):")
    for g, d in out["debilidades"].items():
        print(f"  {g}: {d['entradas_carril']}  | xG concedido por carril/partido: {d['xg_carril_partido']}")
    print("\nDe donde viene el xG que concede (por partido):")
    for g, d in out["debilidades"].items():
        print(f"  {g}: " + ", ".join(f"{o} {v['xg_partido']} ({v['goles']} goles)" for o, v in list(d["origen"].items())[:5]))


if __name__ == "__main__":
    main()
