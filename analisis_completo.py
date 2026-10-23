"""
Analisis completo (Hackathon ISAC 2026): todo lo que faltaba del reto.

  5.1  Organizacion defensiva (360: altura y compacidad del bloque) y transiciones
  5.2  Variabilidad por contexto: marcador, localia y fuerza del rival
  5.3  Uso de jugadores: formaciones, red de pases, jugadores clave, sustituciones
  5.4  Balon parado ofensivo y defensivo (zonas de envio y remate, marcaje con 360)
  5.5  OBV por tipo de accion y por jugador
  Sim  Monte Carlo con xG: que tan merecido fue cada resultado

Entradas (en data/): partidos.csv, huella.csv, xt_grid.npy, events/, frames/, lineups/
Salida:  data/complementos.json   (solo resultados agregados)

Uso (misma carpeta que construir_huella.py):
    pip install networkx
    python analisis_completo.py
Usa SB_USERNAME / SB_PASSWORD (si existen) solo para leer el calendario y calcular la
fuerza de cada rival; sin credenciales usa datos abiertos o se salta ese paso.
"""
import argparse
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from construir_huella import (TERCIO_FINAL, XT_NX, XT_NY, celda, col, es_juego_abierto,
                              leer, preparar, xy)

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(2026)

FIN_TRI = pd.Timestamp("2024-12-16")
VENTANA_TRANS = 10.0     # segundos, igual que "transicion rapida" del framework
VENTANA_RECUP = 5.0      # segundos para "recuperacion inmediata"
VENTANA_CAMBIO = 15.0    # minutos antes / despues de una sustitucion
N_SIM = 20000
JUEGO_ABIERTO = {"Regular Play", "From Counter"}
PATRON_BP = {"From Corner": "Corner", "From Free Kick": "Tiro libre", "From Throw In": "Saque lateral"}

ROL = {
    "Goalkeeper": "POR",
    "Right Back": "LD", "Right Wing Back": "LD", "Left Back": "LI", "Left Wing Back": "LI",
    "Right Center Back": "DCD", "Left Center Back": "DCI", "Center Back": "DC",
    "Right Defensive Midfield": "MCD", "Center Defensive Midfield": "MCD", "Left Defensive Midfield": "MCD",
    "Right Center Midfield": "MC", "Center Midfield": "MC", "Left Center Midfield": "MC",
    "Right Midfield": "ED", "Right Wing": "ED", "Left Midfield": "EI", "Left Wing": "EI",
    "Right Attacking Midfield": "MCO", "Center Attacking Midfield": "MCO", "Left Attacking Midfield": "MCO",
    "Secondary Striker": "MCO",
    "Right Center Forward": "DEL", "Striker": "DEL", "Left Center Forward": "DEL",
}
NOMBRE_ROL = {"POR": "Portero", "LD": "Lateral derecho", "LI": "Lateral izquierdo", "DCD": "Central derecho",
              "DCI": "Central izquierdo", "DC": "Central", "MCD": "Contencion", "MC": "Medio centro",
              "ED": "Extremo derecho", "EI": "Extremo izquierdo", "MCO": "Mediapunta", "DEL": "Delantero"}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def r(v, d=4):
    if v is None:
        return None
    try:
        if np.isnan(v):
            return None
    except TypeError:
        pass
    return round(float(v), d)


def boot(vals, n=1000):
    v = np.asarray([x for x in vals if x is not None and not np.isnan(x)], float)
    if len(v) < 3:
        return (r(v.mean()) if len(v) else None), None, None
    m = v[RNG.integers(0, len(v), (n, len(v)))].mean(1)
    return r(v.mean()), r(np.percentile(m, 5)), r(np.percentile(m, 95))


def etapa(caso, fecha):
    if caso != "Jardine-America":
        return None
    return "Tricampeonato" if pd.Timestamp(fecha) <= FIN_TRI else "Despues"


def goles_tiempos(ev, equipo):
    g = ev[((ev["type"] == "Shot") & (col(ev, "shot_outcome") == "Goal")) | (ev["type"] == "Own Goal For")]
    return np.sort(g.loc[g["team"] == equipo, "t_abs"].values), np.sort(g.loc[g["team"] != equipo, "t_abs"].values)


def estado(ev, equipo):
    """Marcador del equipo (+1 ganando, 0 empatando, -1 perdiendo) antes de cada evento."""
    a, b = goles_tiempos(ev, equipo)
    t = ev["t_abs"].values
    return np.sign(np.searchsorted(a, t, "left") - np.searchsorted(b, t, "left"))


def minutos_estado(ev, equipo):
    a, b = goles_tiempos(ev, equipo)
    cortes = np.sort(np.concatenate([a, b, [0.0, ev.attrs["duracion"]]]))
    out = {1: 0.0, 0: 0.0, -1: 0.0}
    for t0, t1 in zip(cortes[:-1], cortes[1:]):
        mid = (t0 + t1) / 2
        s = int(np.sign(np.searchsorted(a, mid) - np.searchsorted(b, mid)))
        out[s] += (t1 - t0) / 60
    return out


def posesiones(ev):
    g = ev.groupby("possession", sort=True)
    p = pd.DataFrame({
        "equipo": g["possession_team"].first(),
        "patron": g["play_pattern"].first(),
        "t0": g["t_abs"].min(),
        "t1": g["t_abs"].max(),
    }).reset_index()
    return p


def npxg(df):
    s = df[(df["type"] == "Shot") & (col(df, "shot_type") != "Penalty")]
    return col(s, "shot_statsbomb_xg").fillna(0).sum()


def xt_movs(ev, xt):
    pa = ev[(ev["type"] == "Pass") & col(ev, "pass_outcome").isna() & es_juego_abierto(ev)
            & ev["x"].notna() & ev["pex"].notna()]
    ca = ev[(ev["type"] == "Carry") & ev["x"].notna() & ev["cex"].notna()]
    m = pd.concat([pa.assign(ex=pa["pex"], ey=pa["pey"]), ca.assign(ex=ca["cex"], ey=ca["cey"])])
    if m.empty:
        return m.assign(dxt=[])
    ix, iy = celda(m["x"].values, m["y"].values, XT_NX, XT_NY)
    jx, jy = celda(m["ex"].values, m["ey"].values, XT_NX, XT_NY)
    m["dxt"] = xt[jx, jy] - xt[ix, iy]
    return m


def formacion_txt(f):
    try:
        return "-".join(str(int(float(f))))
    except (TypeError, ValueError):
        return None


def tactica(v):
    if isinstance(v, dict):
        return v
    try:
        return dict(v)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 5.2 Contexto
# ---------------------------------------------------------------------------
def contexto_partido(ev, equipo):
    """Numeradores y denominadores por estado del marcador, para agregar sin sesgo."""
    st = estado(ev, equipo)
    mins = minutos_estado(ev, equipo)
    out = {}
    eq = ev["team"] == equipo
    for s in (1, 0, -1):
        m = st == s
        e = ev[m]
        q, rv = e[e["team"] == equipo], e[e["team"] != equipo]
        pc_q = ((q["type"] == "Pass") & col(q, "pass_outcome").isna()).sum()
        pc_r = ((rv["type"] == "Pass") & col(rv, "pass_outcome").isna()).sum()
        ft_q = ((q["type"] == "Pass") & (q["x"] >= TERCIO_FINAL)).sum()
        ft_r = ((rv["type"] == "Pass") & (rv["x"] >= TERCIO_FINAL)).sum()
        ppda_n = ((rv["type"] == "Pass") & (rv["x"] < 72)).sum()
        ppda_d = ((q["x"] >= 48) & (q["type"].isin(["Interception", "Foul Committed"])
                                     | ((q["type"] == "Duel") & (col(q, "duel_type") == "Tackle")))).sum()
        pres = q[q["type"] == "Pressure"]
        out[s] = dict(min=mins[s], xg=npxg(q), xga=npxg(rv), pc_q=pc_q, pc_r=pc_r, ft_q=ft_q, ft_r=ft_r,
                      ppda_n=ppda_n, ppda_d=ppda_d, pres=len(pres), pres_alta=(pres["x"] >= TERCIO_FINAL).sum())
    return out


def agregar_contexto(filas):
    """filas: lista de dicts por partido y estado. Devuelve tasas con IC por bootstrap de partidos."""
    df = pd.DataFrame(filas)
    res = []
    for (caso, s), g in df.groupby(["caso", "estado"]):
        g = g[g["min"] > 0]
        if g["min"].sum() < 30:
            continue

        def tasa(d):
            m = d["min"].sum()
            return dict(
                xg90=d["xg"].sum() / m * 90, xga90=d["xga"].sum() / m * 90,
                posesion=d["pc_q"].sum() / max(1, d["pc_q"].sum() + d["pc_r"].sum()),
                field_tilt=d["ft_q"].sum() / max(1, d["ft_q"].sum() + d["ft_r"].sum()),
                ppda=d["ppda_n"].sum() / max(1, d["ppda_d"].sum()),
                presion_alta=d["pres_alta"].sum() / max(1, d["pres"].sum()))
        base = tasa(g)
        idx = np.arange(len(g))
        bs = [tasa(g.iloc[RNG.choice(idx, len(idx))]) for _ in range(400)]
        row = {"caso": caso, "estado": {1: "Ganando", 0: "Empatando", -1: "Perdiendo"}[s],
               "minutos": r(g["min"].sum(), 1), "partidos": int(g["match_id"].nunique())}
        for k, v in base.items():
            vals = [b[k] for b in bs]
            row[k] = r(v)
            row[k + "_lo"], row[k + "_hi"] = r(np.percentile(vals, 5)), r(np.percentile(vals, 95))
        res.append(row)
    return res


def fuerza_rivales(partidos, competition_id):
    """Puntos por partido de cada equipo en cada temporada -> tercil Fuerte/Medio/Debil."""
    try:
        from statsbombpy import sb
        creds = None
        if os.environ.get("SB_USERNAME"):
            creds = {"user": os.environ["SB_USERNAME"], "passwd": os.environ["SB_PASSWORD"]}
        kw = {"creds": creds} if creds else {}
        comps = sb.competitions(**kw)
        comps = comps[comps["competition_id"] == competition_id]
        temporadas = set(partidos["season"].astype(str))
        tablas = []
        for row in comps.itertuples():
            if str(row.season_name) not in temporadas:
                continue
            m = sb.matches(competition_id=competition_id, season_id=row.season_id, **kw)
            m = m.dropna(subset=["home_score", "away_score"])
            for lado, otro in (("home", "away"), ("away", "home")):
                pts = np.where(m[f"{lado}_score"] > m[f"{otro}_score"], 3,
                               np.where(m[f"{lado}_score"] == m[f"{otro}_score"], 1, 0))
                tablas.append(pd.DataFrame({"season": str(row.season_name), "equipo": m[f"{lado}_team"], "pts": pts}))
        t = pd.concat(tablas).groupby(["season", "equipo"])["pts"].mean().reset_index(name="ppg")
        t["tercil"] = t.groupby("season")["ppg"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 3, labels=["Debil", "Medio", "Fuerte"]))
        return {(a, b): c for a, b, c in zip(t["season"], t["equipo"], t["tercil"].astype(str))}
    except Exception as e:
        print(f"  (fuerza de rivales no disponible: {e})")
        return {}


# ---------------------------------------------------------------------------
# 5.1 Organizacion defensiva (360) y transiciones
# ---------------------------------------------------------------------------
def bloque_defensivo(ev, fr, equipo):
    """Altura, linea y compacidad del bloque cuando el rival tiene el balon en juego abierto."""
    if fr is None or fr.empty:
        return None
    rv = ev[(ev["team"] != equipo) & ev["type"].isin(["Pass", "Carry"])
            & col(ev, "play_pattern").isin(JUEGO_ABIERTO | {"From Goal Kick", "From Keeper", "From Kick Off"})]
    if rv.empty:
        return None
    f = fr[fr["id"].isin(rv["id"])].copy()
    f["fx"], f["fy"] = xy(f["location"])
    nos = f[(f["teammate"] == False) & (f["keeper"] == False)].copy()
    nos["xo"] = 120 - nos["fx"]
    nos["yo"] = 80 - nos["fy"]
    act = f[f["actor"] == True][["id", "fx", "fy"]].rename(columns={"fx": "ax", "fy": "ay"})
    nos = nos.merge(act, on="id", how="left")
    nos["dist"] = np.hypot(nos["fx"] - nos["ax"], nos["fy"] - nos["ay"])
    nos["cerca5"] = (nos["dist"] <= 5).astype(int)
    g = nos.groupby("id")
    t = pd.DataFrame({
        "n": g.size(),
        "altura": g["xo"].mean(),
        "linea": nos.sort_values("xo").groupby("id").head(4).groupby("id")["xo"].mean(),
        "largo": g["xo"].max() - g["xo"].min(),
        "ancho": g["yo"].max() - g["yo"].min(),
        "cerca": g["cerca5"].sum(),
    })
    t = t[t["n"] >= 7]
    if t.empty:
        return None
    bx = rv.set_index("id")["x"]
    t["bola_o"] = 120 - bx.reindex(t.index).values
    t["zona"] = pd.cut(t["bola_o"], [-1, 40, 80, 121], labels=["Bloque bajo", "Medio", "Presion"])
    out = {"frames": int(len(t))}
    for z in ["Presion", "Medio", "Bloque bajo", "Total"]:
        s = t if z == "Total" else t[t["zona"] == z]
        if len(s) >= 10:
            out[z] = {k: float(s[k].mean()) for k in ["altura", "linea", "largo", "ancho", "cerca"]}
    return out


def transiciones(ev, equipo, xg_por_tiro):
    """Perdidas y recuperaciones en juego abierto y lo que pasa en los 10 s siguientes."""
    p = posesiones(ev)
    tiros = ev[ev["type"] == "Shot"][["team", "t_abs", "possession"]].copy()
    tiros["xg"] = col(ev[ev["type"] == "Shot"], "shot_statsbomb_xg").fillna(0).values
    out = defaultdict(float)
    for i in range(len(p) - 1):
        a, b = p.iloc[i], p.iloc[i + 1]
        if a["equipo"] == b["equipo"] or b["patron"] not in JUEGO_ABIERTO:
            continue
        lado = "def" if a["equipo"] == equipo else "of"
        out[f"{lado}_n"] += 1
        ts = tiros[(tiros["possession"] == b["possession"]) & (tiros["t_abs"] - b["t0"] <= VENTANA_TRANS)]
        out[f"{lado}_tiros"] += len(ts)
        out[f"{lado}_xg"] += ts["xg"].sum()
        if lado == "def" and i + 2 < len(p):
            c = p.iloc[i + 2]
            if c["equipo"] == equipo and c["t0"] - b["t0"] <= VENTANA_RECUP:
                out["def_recup5"] += 1
    out["min"] = ev.attrs["duracion"] / 60
    return dict(out)


# ---------------------------------------------------------------------------
# 5.3 Jugadores
# ---------------------------------------------------------------------------
def formaciones(ev, equipo):
    q = ev[(ev["team"] == equipo) & ev["type"].isin(["Starting XI", "Tactical Shift"])].sort_values("t_abs")
    if q.empty:
        return {}
    marcas = [(row.t_abs, formacion_txt(tactica(row.tactics).get("formation"))) for row in q.itertuples()]
    marcas.append((ev.attrs["duracion"], None))
    mins = defaultdict(float)
    for (t0, f), (t1, _) in zip(marcas[:-1], marcas[1:]):
        if f:
            mins[f] += (t1 - t0) / 60
    return dict(mins)


def minutos_jugador(lu, equipo, dur_min):
    """Minutos por jugador a partir de las alineaciones (posiciones con from/to)."""
    if lu is None or lu.empty:
        return {}
    out = {}
    for row in lu[lu["team"] == equipo].itertuples():
        pos = row.positions if isinstance(row.positions, (list, np.ndarray)) else []
        tot = 0.0
        for p in pos:
            p = dict(p)

            def mm(s):
                if s is None or (isinstance(s, float) and np.isnan(s)):
                    return None
                a, b = str(s).split(":")[:2]
                return int(a) + int(b) / 60
            a, b = mm(p.get("from")), mm(p.get("to"))
            tot += max(0.0, (b if b is not None else dur_min) - (a or 0.0))
        if tot > 0:
            out[row.player_name] = tot
    return out


def red_pases(ev, equipo):
    q = ev[ev["team"] == equipo]
    rol_j = q.dropna(subset=["position"]).groupby("player")["position"].agg(lambda s: s.mode().iloc[0]).map(ROL)
    pa = q[(q["type"] == "Pass") & col(q, "pass_outcome").isna() & es_juego_abierto(q)].copy()
    pa["r_de"] = pa["player"].map(rol_j)
    pa["r_a"] = col(pa, "pass_recipient").map(rol_j)
    nodos = pa.groupby("r_de").agg(n=("x", "size"), x=("x", "mean"), y=("y", "mean"))
    aristas = pa.dropna(subset=["r_de", "r_a"])
    aristas = aristas[aristas["r_de"] != aristas["r_a"]].groupby(["r_de", "r_a"]).size()
    return nodos, aristas


def centralidad(aristas):
    """PageRank ponderado y betweenness (networkx, distancia = 1 / pases)."""
    if aristas.empty:
        return {}, {}
    try:
        import networkx as nx
        G = nx.DiGraph()
        for (a, b), w in aristas.items():
            G.add_edge(a, b, weight=float(w), dist=1.0 / float(w))
        bt = nx.betweenness_centrality(G, weight="dist", normalized=True)
        try:
            pr = nx.pagerank(G, weight="weight")
        except ImportError:  # networkx 3.x usa scipy para PageRank
            pr = None
        if pr is not None:
            return pr, bt
        return centralidad_numpy(aristas), bt
    except ImportError:
        return centralidad_numpy(aristas), {}


def centralidad_numpy(aristas):
    """PageRank ponderado con iteracion de potencias (sin scipy)."""
    if True:
        nodos = sorted(set(a for a, _ in aristas.index) | set(b for _, b in aristas.index))
        k = {n: i for i, n in enumerate(nodos)}
        M = np.zeros((len(nodos), len(nodos)))
        for (a, b), w in aristas.items():
            M[k[a], k[b]] = w
        M = M / np.where(M.sum(1, keepdims=True) == 0, 1, M.sum(1, keepdims=True))
        v = np.ones(len(nodos)) / len(nodos)
        for _ in range(100):
            v = 0.15 / len(nodos) + 0.85 * v @ M
        return {n: float(v[k[n]]) for n in nodos}


def ventanas_cambio(ev, equipo, xt):
    """Dominio (parte del xT) y xG en los 15 min antes y despues de cada ventana de cambios."""
    subs = ev[(ev["type"] == "Substitution") & (ev["team"] == equipo)]["t_abs"].sort_values().values / 60
    movs = xt_movs(ev, xt)
    movs["m"] = movs["t_abs"] / 60
    tiros = ev[(ev["type"] == "Shot") & (col(ev, "shot_type") != "Penalty")].copy()
    tiros["m"] = tiros["t_abs"] / 60
    tiros["xg"] = col(tiros, "shot_statsbomb_xg").fillna(0)
    dur = ev.attrs["duracion"] / 60

    def medir(t0, t1):
        mm = movs[(movs["m"] >= t0) & (movs["m"] < t1)]
        a = mm.loc[mm["team"] == equipo, "dxt"].clip(lower=0).sum()
        b = mm.loc[mm["team"] != equipo, "dxt"].clip(lower=0).sum()
        tt = tiros[(tiros["m"] >= t0) & (tiros["m"] < t1)]
        return (a / (a + b) if a + b > 0 else np.nan,
                tt.loc[tt["team"] == equipo, "xg"].sum(), tt.loc[tt["team"] != equipo, "xg"].sum())

    ventanas, ultimo = [], -99
    for t in subs:
        if t - ultimo > 2:
            ventanas.append(t)
        ultimo = t
    res = []
    for t in ventanas:
        if t < 46 or t > dur - VENTANA_CAMBIO:
            continue
        d0, x0, c0 = medir(t - VENTANA_CAMBIO, t)
        d1, x1, c1 = medir(t, t + VENTANA_CAMBIO)
        res.append({"tipo": "cambio", "min": t, "d_dom": d1 - d0, "d_xg": x1 - x0, "d_xga": c1 - c0})
    # Control: minutos del segundo tiempo sin cambios a +-15 min
    for t in np.arange(50, dur - VENTANA_CAMBIO, 5.0):
        if len(subs) and np.min(np.abs(subs - t)) < VENTANA_CAMBIO:
            continue
        d0, x0, c0 = medir(t - VENTANA_CAMBIO, t)
        d1, x1, c1 = medir(t, t + VENTANA_CAMBIO)
        res.append({"tipo": "control", "min": t, "d_dom": d1 - d0, "d_xg": x1 - x0, "d_xga": c1 - c0})
    primer = float(subs[0]) if len(subs) else None
    return res, primer, int(len(subs))


# ---------------------------------------------------------------------------
# 5.4 Balon parado
# ---------------------------------------------------------------------------
def zona_area(x, y, y_corner):
    if np.isnan(x) or np.isnan(y):
        return None
    if x < 102 or y < 18 or y > 62:
        return "Fuera del area"
    yr = y if (y_corner is None or y_corner < 40) else 80 - y
    ancho = "Primer palo" if yr < 36 else ("Centro" if yr <= 44 else "Segundo palo")
    fondo = "Area chica" if x >= 114 else ("Punto penal" if x >= 108 else "Frontal")
    return f"{fondo} | {ancho}"


def balon_parado(ev, fr, equipo):
    out = {"of": defaultdict(float), "def": defaultdict(float)}
    zonas = {"of": {"envio": defaultdict(float), "remate": defaultdict(lambda: [0, 0.0])},
             "def": {"envio": defaultdict(float), "remate": defaultdict(lambda: [0, 0.0])}}
    tecnica = {"of": defaultdict(float), "def": defaultdict(float)}
    marcaje = defaultdict(float)
    p = posesiones(ev)
    for row in p.itertuples():
        tipo = PATRON_BP.get(row.patron)
        if not tipo:
            continue
        e = ev[ev["possession"] == row.possession]
        lado = "of" if row.equipo == equipo else "def"
        ini = e[e["type"] == "Pass"].head(1)
        x0 = float(ini["x"].iloc[0]) if len(ini) else np.nan
        if tipo == "Tiro libre" and not (x0 >= 60):
            continue
        if tipo == "Saque lateral" and not (x0 >= TERCIO_FINAL):
            continue
        tiros = e[(e["type"] == "Shot") & (e["team"] == row.equipo)]
        xg = col(tiros, "shot_statsbomb_xg").fillna(0).sum()
        out[lado][f"{tipo}|n"] += 1
        out[lado][f"{tipo}|tiro"] += int(len(tiros) > 0)
        out[lado][f"{tipo}|xg"] += xg
        out[lado][f"{tipo}|gol"] += int((col(tiros, "shot_outcome") == "Goal").sum())
        if tipo == "Corner" and len(ini):
            yc = float(ini["y"].iloc[0])
            z = zona_area(float(ini["pex"].iloc[0]), float(ini["pey"].iloc[0]), yc)
            if z:
                zonas[lado]["envio"][z] += 1
            tec = col(ini, "pass_technique").iloc[0]
            tecnica[lado][tec if isinstance(tec, str) else "Sin dato"] += 1
            for t in tiros.itertuples():
                z = zona_area(t.x, t.y, yc)
                if z:
                    zonas[lado]["remate"][z][0] += 1
                    zonas[lado]["remate"][z][1] += float(getattr(t, "shot_statsbomb_xg", 0) or 0)
            # 360: quien esta en el area al momento del envio
            if fr is not None and not fr.empty:
                f = fr[fr["id"] == ini["id"].iloc[0]].copy()
                if len(f) >= 8:
                    f["fx"], f["fy"] = xy(f["location"])
                    atac = f[(f["teammate"] == True) & (f["actor"] == False)]
                    defe = f[(f["teammate"] == False) & (f["keeper"] == False)]
                    en_area = lambda d: d[(d["fx"] >= 102) & (d["fy"] >= 18) & (d["fy"] <= 62)]
                    da, aa = en_area(defe), en_area(atac)
                    k = "nos_atacando" if lado == "of" else "nos_defendiendo"
                    marcaje[f"{k}|frames"] += 1
                    marcaje[f"{k}|atac_area"] += len(aa)
                    marcaje[f"{k}|def_area"] += len(da)
                    if lado == "def":
                        marcaje["def|chica"] += ((da["fx"] >= 114) & (da["fy"] >= 30) & (da["fy"] <= 50)).sum()
                        if len(da) and len(aa):
                            dmin = np.hypot(da["fx"].values[:, None] - aa["fx"].values[None, :],
                                            da["fy"].values[:, None] - aa["fy"].values[None, :]).min(1)
                            marcaje["def|individual"] += (dmin <= 2.0).sum()
                            marcaje["def|en_area_eval"] += len(dmin)
    return out, zonas, tecnica, marcaje


# ---------------------------------------------------------------------------
# Simulacion Monte Carlo
# ---------------------------------------------------------------------------
def simular(ev, equipo):
    tiros = ev[ev["type"] == "Shot"].copy()
    tiros["xg"] = col(tiros, "shot_statsbomb_xg").fillna(0)
    p = tiros.groupby(["team", "possession"])["xg"].apply(lambda s: 1 - np.prod(1 - s.values)).reset_index()
    pa = p.loc[p["team"] == equipo, "xg"].values
    pb = p.loc[p["team"] != equipo, "xg"].values
    ga = (RNG.random((N_SIM, len(pa))) < pa).sum(1) if len(pa) else np.zeros(N_SIM)
    gb = (RNG.random((N_SIM, len(pb))) < pb).sum(1) if len(pb) else np.zeros(N_SIM)
    pw, pd_, pl = (ga > gb).mean(), (ga == gb).mean(), (ga < gb).mean()
    return dict(xg=float(tiros.loc[tiros["team"] == equipo, "xg"].sum()),
                xga=float(tiros.loc[tiros["team"] != equipo, "xg"].sum()),
                pw=float(pw), pd=float(pd_), pl=float(pl), xpts=float(3 * pw + pd_))


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--competition", type=int, default=73)
    args = ap.parse_args()
    data = Path(args.data)

    partidos = pd.read_csv(data / "partidos.csv")
    huella = pd.read_csv(data / "huella.csv")
    xt = np.load(data / "xt_grid.npy")
    hay_obv = None

    ctx_filas, def_filas, trans_filas, form_filas, sub_filas, sim_filas = [], [], [], [], [], []
    redes = defaultdict(lambda: {"nodos": [], "aristas": defaultdict(float), "partidos": 0})
    jug = defaultdict(lambda: defaultdict(float))
    jug_rol = defaultdict(lambda: defaultdict(float))
    obv_tipo = defaultdict(lambda: defaultdict(float))
    bp_tot = defaultdict(lambda: {"of": defaultdict(float), "def": defaultdict(float)})
    bp_zonas = defaultdict(lambda: {s: {"envio": defaultdict(float), "remate": defaultdict(lambda: [0, 0.0])}
                                    for s in ("of", "def")})
    bp_tec = defaultdict(lambda: {"of": defaultdict(float), "def": defaultdict(float)})
    bp_marca = defaultdict(lambda: defaultdict(float))
    min_caso = defaultdict(float)
    timing = defaultdict(list)

    filas = list(partidos.sort_values("match_date").itertuples())
    print(f"Procesando {len(filas)} partidos...")
    for k, rw in enumerate(filas, 1):
        mid, caso, equipo = int(rw.match_id), rw.caso, rw.equipo
        ev = leer(data / "events" / str(mid))
        if ev is None or ev.empty:
            continue
        ev = preparar(ev)
        fr = leer(data / "frames" / str(mid))
        lu = leer(data / "lineups" / str(mid))
        dur = ev.attrs["duracion"] / 60
        et = etapa(caso, rw.match_date)
        grupos = [caso] + ([f"{caso}|{et}"] if et else [])
        min_caso[caso] += dur

        # 5.2 contexto por marcador
        for s, d in contexto_partido(ev, equipo).items():
            ctx_filas.append({"caso": caso, "match_id": mid, "estado": s, **d})
            if et:
                ctx_filas.append({"caso": f"{caso}|{et}", "match_id": mid, "estado": s, **d})

        # 5.1 bloque defensivo y transiciones
        b = bloque_defensivo(ev, fr, equipo)
        if b:
            for g in grupos:
                def_filas.append({"caso": g, "match_id": mid, **b})
        tr = transiciones(ev, equipo, None)
        for g in grupos:
            trans_filas.append({"caso": g, "match_id": mid, **tr})

        # 5.3 formaciones, red de pases, jugadores, cambios
        for f, m in formaciones(ev, equipo).items():
            for g in grupos:
                form_filas.append({"caso": g, "formacion": f, "min": m})
        nodos, aristas = red_pases(ev, equipo)
        for g in grupos:
            redes[g]["partidos"] += 1
            redes[g]["nodos"].append(nodos)
            for key, w in aristas.items():
                redes[g]["aristas"][key] += w
        mins = minutos_jugador(lu, equipo, dur)
        q = ev[ev["team"] == equipo]
        mv = xt_movs(q, xt)
        xt_j = mv.groupby("player")["dxt"].apply(lambda s: s.clip(lower=0).sum())
        pa = q[(q["type"] == "Pass") & col(q, "pass_outcome").isna() & es_juego_abierto(q) & q["pex"].notna()]
        d0 = np.hypot(120 - pa["x"], 40 - pa["y"])
        d1 = np.hypot(120 - pa["pex"], 40 - pa["pey"])
        prog_j = pa[d1 <= 0.75 * d0].groupby("player").size()
        if hay_obv is None:
            hay_obv = "obv_total_net" in ev.columns
        obv_j = q.groupby("player")["obv_total_net"].sum() if hay_obv else pd.Series(dtype=float)
        if hay_obv:
            for t, v in q.groupby("type")["obv_total_net"].sum().items():
                if v != 0:
                    obv_tipo[caso][t] += float(v)
                    if et:
                        obv_tipo[f"{caso}|{et}"][t] += float(v)
            rvq = ev[ev["team"] != equipo]
            for t, v in rvq.groupby("type")["obv_total_net"].sum().items():
                obv_tipo["Liga"][t] += float(v)
            obv_tipo["Liga"]["__min"] += dur
            obv_tipo[caso]["__min"] += dur
            if et:
                obv_tipo[f"{caso}|{et}"]["__min"] += dur
        for jn, m in mins.items():
            jug[(caso, jn)]["min"] += m
            jug[(caso, jn)]["partidos"] += 1
            jug[(caso, jn)]["xt"] += float(xt_j.get(jn, 0))
            jug[(caso, jn)]["prog"] += float(prog_j.get(jn, 0))
            jug[(caso, jn)]["obv"] += float(obv_j.get(jn, 0)) if hay_obv else 0
        for jn, pos in q.dropna(subset=["position"]).groupby("player")["position"].agg(lambda s: s.mode().iloc[0]).items():
            jug_rol[(caso, jn)][ROL.get(pos, pos)] += 1
        res, primer, nsub = ventanas_cambio(ev, equipo, xt)
        for x in res:
            for g in grupos:
                sub_filas.append({"caso": g, "match_id": mid, **x})
        for g in grupos:
            timing[g].append((primer, nsub))

        # 5.4 balon parado
        tot, zon, tec, mar = balon_parado(ev, fr, equipo)
        for g in grupos:
            for lado in ("of", "def"):
                for kk, v in tot[lado].items():
                    bp_tot[g][lado][kk] += v
                for kk, v in zon[lado]["envio"].items():
                    bp_zonas[g][lado]["envio"][kk] += v
                for kk, (n, x) in zon[lado]["remate"].items():
                    bp_zonas[g][lado]["remate"][kk][0] += n
                    bp_zonas[g][lado]["remate"][kk][1] += x
                for kk, v in tec[lado].items():
                    bp_tec[g][lado][kk] += v
            for kk, v in mar.items():
                bp_marca[g][kk] += v
            bp_tot[g]["of"]["__partidos"] += 1

        # Simulacion
        s = simular(ev, equipo)
        ga, gr = int(rw.goles_equipo), int(rw.goles_rival)
        pts = 3 if ga > gr else (1 if ga == gr else 0)
        sim_filas.append({"caso": caso, "etapa": et, "match_id": mid, "fecha": str(rw.match_date), "rival": rw.rival,
                          "fase": str(getattr(rw, "competition_stage", "")), "goles": [ga, gr], "pts": pts, **s})
        if k % 50 == 0:
            print(f"  {k}/{len(filas)}")

    salida = {"hay_obv": bool(hay_obv)}

    # 5.2 -------------------------------------------------------------
    salida["contexto_marcador"] = agregar_contexto(ctx_filas)
    hp = huella.merge(partidos[["match_id", "equipo", "es_local", "season"]], on=["match_id", "equipo"], how="inner")
    hp["season"] = hp["season_y"] if "season_y" in hp.columns else hp["season"]
    fuerza = fuerza_rivales(partidos, args.competition)
    hp["tercil_rival"] = [fuerza.get((str(s), rv)) for s, rv in zip(hp["season"], hp["rival"])]
    pts = {(x["match_id"], x["caso"]): x["pts"] for x in sim_filas}
    hp["pts"] = [pts.get((m, c)) for m, c in zip(hp["match_id"], hp["caso"])]
    kpis = ["posesion_pct", "field_tilt", "npxg_p90", "npxg_contra_p90", "ppda", "presion_alta_pct", "pts"]
    ctx2 = []
    for var, etiqueta in (("es_local", {True: "Local", False: "Visitante"}), ("tercil_rival", None)):
        if var not in hp or hp[var].isna().all():
            continue
        for (caso, v), g in hp.dropna(subset=[var]).groupby(["caso", var]):
            row = {"caso": caso, "variable": "Localia" if var == "es_local" else "Rival",
                   "valor": etiqueta[v] if etiqueta else v, "partidos": int(len(g))}
            for kp in kpis:
                m, lo, hi = boot(g[kp].astype(float).values)
                row[kp], row[kp + "_lo"], row[kp + "_hi"] = m, lo, hi
            ctx2.append(row)
    salida["contexto_partido"] = ctx2

    # 5.1 -------------------------------------------------------------
    blo = []
    df = pd.DataFrame(def_filas)
    for caso, g in df.groupby("caso") if len(df) else []:
        for z in ["Presion", "Medio", "Bloque bajo", "Total"]:
            vals = [x for x in g[z] if isinstance(x, dict)] if z in g else []
            if len(vals) < 3:
                continue
            row = {"caso": caso, "zona": z, "partidos": len(vals)}
            for kk in ["altura", "linea", "largo", "ancho", "cerca"]:
                row[kk], row[kk + "_lo"], row[kk + "_hi"] = boot([v[kk] for v in vals])
            blo.append(row)
    salida["bloque_defensivo"] = blo
    tr = pd.DataFrame(trans_filas).fillna(0)
    tlist = []
    for caso, g in tr.groupby("caso"):
        per = lambda num, den: (g[num].sum() / g[den].sum()) if g[den].sum() else None
        mn = g["min"].sum()
        tlist.append({"caso": caso, "partidos": int(len(g)),
                      "perdidas_p90": r(g["def_n"].sum() / mn * 90, 2),
                      "recup5_pct": r(per("def_recup5", "def_n")),
                      "tiros_rival_por100": r(per("def_tiros", "def_n") * 100 if per("def_tiros", "def_n") is not None else None, 2),
                      "xg_rival_trans_p90": r(g["def_xg"].sum() / mn * 90, 3),
                      "recuperaciones_p90": r(g["of_n"].sum() / mn * 90, 2),
                      "tiros_propios_por100": r(per("of_tiros", "of_n") * 100 if per("of_tiros", "of_n") is not None else None, 2),
                      "xg_propio_trans_p90": r(g["of_xg"].sum() / mn * 90, 3)})
    salida["transiciones"] = tlist

    # 5.3 -------------------------------------------------------------
    fd = pd.DataFrame(form_filas)
    salida["formaciones"] = []
    if len(fd):
        for caso, g in fd.groupby("caso"):
            t = g.groupby("formacion")["min"].sum().sort_values(ascending=False)
            t = t / t.sum()
            salida["formaciones"].append({"caso": caso, "reparto": [[f, r(v)] for f, v in t.head(6).items()]})
    salida["redes"] = {}
    for g, d in redes.items():
        if not d["nodos"]:
            continue
        nd = pd.concat(d["nodos"]).reset_index().rename(columns={"r_de": "rol"})
        nd["sx"], nd["sy"] = nd["x"] * nd["n"], nd["y"] * nd["n"]
        nd = nd.groupby("rol").agg(n=("n", "sum"), sx=("sx", "sum"), sy=("sy", "sum"))
        nd["x"], nd["y"] = nd["sx"] / nd["n"], nd["sy"] / nd["n"]
        nd["n90"] = nd["n"] / d["partidos"]
        nd = nd[nd["n90"] >= 3].sort_values("n90", ascending=False).head(12)
        ar = pd.Series(d["aristas"])
        ar = ar[[a in nd.index and b in nd.index for a, b in ar.index]] / d["partidos"]
        pr, bt = centralidad(ar)
        salida["redes"][g] = {
            "partidos": d["partidos"],
            "nodos": [{"rol": k2, "nombre": NOMBRE_ROL.get(k2, k2), "x": r(v.x, 1), "y": r(v.y, 1),
                       "pases": r(v.n90, 1), "pagerank": r(pr.get(k2)), "intermediacion": r(bt.get(k2))}
                      for k2, v in nd.iterrows()],
            "aristas": [[a, b, r(w, 2)] for (a, b), w in ar.sort_values(ascending=False).items() if w >= 1],
        }
    jl = []
    for (caso, jn), d in jug.items():
        if d["min"] < max(450, 0.15 * min_caso[caso]):
            continue
        rol = max(jug_rol[(caso, jn)].items(), key=lambda kv: kv[1])[0] if jug_rol[(caso, jn)] else None
        jl.append({"caso": caso, "jugador": jn, "rol": NOMBRE_ROL.get(rol, rol), "minutos": int(d["min"]),
                   "partidos": int(d["partidos"]), "xt90": r(d["xt"] / d["min"] * 90, 3),
                   "prog90": r(d["prog"] / d["min"] * 90, 2),
                   "obv90": r(d["obv"] / d["min"] * 90, 3) if hay_obv else None})
    salida["jugadores"] = sorted(jl, key=lambda x: -(x["obv90"] if x["obv90"] is not None else x["xt90"]))
    sd = pd.DataFrame(sub_filas)
    sl = []
    if len(sd):
        for caso, g in sd.groupby("caso"):
            c, k0 = g[g["tipo"] == "cambio"], g[g["tipo"] == "control"]
            if len(c) < 3:
                continue
            row = {"caso": caso, "ventanas": int(len(c))}
            for m in ["d_dom", "d_xg", "d_xga"]:
                row[m], row[m + "_lo"], row[m + "_hi"] = boot(c[m].values)
                row[m + "_control"] = r(k0[m].mean()) if len(k0) else None
                difs = []
                cv, kv = c[m].dropna().values, k0[m].dropna().values
                for _ in range(800):
                    if len(cv) and len(kv):
                        difs.append(RNG.choice(cv, len(cv)).mean() - RNG.choice(kv, len(kv)).mean())
                row[m + "_neto"] = r(np.mean(difs)) if difs else None
                row[m + "_neto_lo"] = r(np.percentile(difs, 5)) if difs else None
                row[m + "_neto_hi"] = r(np.percentile(difs, 95)) if difs else None
            tm = [t for t in timing[caso] if t[0] is not None]
            row["primer_cambio_min"] = r(np.mean([t[0] for t in tm]), 1) if tm else None
            row["cambios_por_partido"] = r(np.mean([t[1] for t in timing[caso]]), 2)
            sl.append(row)
    salida["sustituciones"] = sl

    # 5.4 -------------------------------------------------------------
    bl = []
    for g, d in bp_tot.items():
        n_part = d["of"].get("__partidos", 0) or 1
        fila = {"caso": g, "partidos": int(n_part), "tipos": {}}
        for lado in ("of", "def"):
            for tipo in ("Corner", "Tiro libre", "Saque lateral"):
                n = d[lado].get(f"{tipo}|n", 0)
                if not n:
                    continue
                fila["tipos"].setdefault(tipo, {})[lado] = {
                    "por_partido": r(n / n_part, 2), "pct_tiro": r(d[lado][f"{tipo}|tiro"] / n),
                    "xg_por_jugada": r(d[lado][f"{tipo}|xg"] / n), "goles": int(d[lado][f"{tipo}|gol"]),
                    "xg_por_partido": r(d[lado][f"{tipo}|xg"] / n_part, 3)}
        z = bp_zonas[g]
        fila["zonas"] = {lado: {"envio": {kk: r(v / max(1, sum(z[lado]["envio"].values()))) for kk, v in z[lado]["envio"].items()},
                                "remate": {kk: [int(n), r(x, 3)] for kk, (n, x) in z[lado]["remate"].items()}}
                         for lado in ("of", "def")}
        fila["tecnica"] = {lado: {kk: r(v / max(1, sum(bp_tec[g][lado].values()))) for kk, v in bp_tec[g][lado].items()}
                           for lado in ("of", "def")}
        m = bp_marca[g]
        fila["marcaje"] = {
            "corners_con_360": int(m.get("nos_defendiendo|frames", 0)),
            "def_en_area": r(m["nos_defendiendo|def_area"] / m["nos_defendiendo|frames"], 2) if m.get("nos_defendiendo|frames") else None,
            "rivales_en_area": r(m["nos_defendiendo|atac_area"] / m["nos_defendiendo|frames"], 2) if m.get("nos_defendiendo|frames") else None,
            "def_area_chica": r(m["def|chica"] / m["nos_defendiendo|frames"], 2) if m.get("nos_defendiendo|frames") else None,
            "pct_individual": r(m["def|individual"] / m["def|en_area_eval"]) if m.get("def|en_area_eval") else None,
            "atacantes_en_area": r(m["nos_atacando|atac_area"] / m["nos_atacando|frames"], 2) if m.get("nos_atacando|frames") else None,
        }
        bl.append(fila)
    salida["balon_parado"] = bl

    # 5.5 -------------------------------------------------------------
    salida["obv"] = []
    if hay_obv:
        for g, d in obv_tipo.items():
            mn = d.get("__min", 0)
            if not mn:
                continue
            salida["obv"].append({"caso": g, "total90": r(sum(v for k2, v in d.items() if k2 != "__min") / mn * 90, 3),
                                  "por_tipo": {k2: r(v / mn * 90, 3) for k2, v in sorted(d.items(), key=lambda kv: -abs(kv[1]))
                                               if k2 != "__min" and abs(v / mn * 90) >= 0.005}})

    # Simulacion --------------------------------------------------------
    sm = pd.DataFrame(sim_filas)
    resumen = []
    for keys, g in list(sm.groupby("caso")) + [((c, e), g) for (c, e), g in sm.dropna(subset=["etapa"]).groupby(["caso", "etapa"])]:
        nombre = keys if isinstance(keys, str) else f"{keys[0]}|{keys[1]}"
        resumen.append({"caso": nombre, "partidos": int(len(g)), "ppg": r(g["pts"].mean(), 2),
                        "xppg": r(g["xpts"].mean(), 2), "dif_ppg": r((g["pts"] - g["xpts"]).mean(), 2),
                        "pts": int(g["pts"].sum()), "xpts": r(g["xpts"].sum(), 1),
                        "victorias": int((g["pts"] == 3).sum()), "victorias_esperadas": r(g["pw"].sum(), 1)})
    salida["simulacion"] = {"resumen": resumen,
                            "partidos": [{k2: (r(v, 3) if isinstance(v, float) else v) for k2, v in x.items()} for x in sim_filas]}

    with open(data / "complementos.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nGuardado: {data / 'complementos.json'} ({(data / 'complementos.json').stat().st_size / 1e6:.2f} MB)")

    # Resumen en pantalla
    pd.set_option("display.width", 200)
    print("\n5.1 Bloque defensivo (altura media del equipo, yardas desde su porteria):")
    if blo:
        b = pd.DataFrame(blo)
        print(b.pivot(index="caso", columns="zona", values="altura").round(1).to_string())
    print("\n5.1 Transiciones:")
    print(pd.DataFrame(tlist).set_index("caso").to_string())
    print("\n5.2 Por marcador (xG a favor por 90 / PPDA):")
    c = pd.DataFrame(salida["contexto_marcador"])
    if len(c):
        print(c.pivot(index="caso", columns="estado", values="xg90").round(2).to_string())
        print(c.pivot(index="caso", columns="estado", values="ppda").round(1).to_string())
    print("\n5.3 Formacion mas usada:")
    for f in salida["formaciones"]:
        print(f"  {f['caso']}: {f['reparto'][:3]}")
    print("\n5.3 Sustituciones (cambio neto de dominio vs control):")
    if sl:
        print(pd.DataFrame(sl)[["caso", "ventanas", "d_dom_neto", "d_dom_neto_lo", "d_dom_neto_hi", "primer_cambio_min"]].to_string(index=False))
    print("\n5.4 Corners a favor (por partido, % con tiro, xG por corner):")
    for f in bl:
        cc = f["tipos"].get("Corner", {}).get("of")
        if cc:
            print(f"  {f['caso']}: {cc['por_partido']} / {cc['pct_tiro']} / {cc['xg_por_jugada']}")
    print("\nSimulacion (puntos por partido reales vs esperados):")
    print(pd.DataFrame(resumen)[["caso", "partidos", "ppg", "xppg", "dif_ppg"]].to_string(index=False))


if __name__ == "__main__":
    main()
