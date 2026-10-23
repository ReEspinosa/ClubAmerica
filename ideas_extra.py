"""
Ideas extra para el reto del Club America (Hackathon ISAC 2026).

1. Transporte optimo (distancia de Wasserstein): cuanto y hacia donde se "mudo"
   el equipo del America de Jardine al de Almada.
2. Control del espacio con 360 (Voronoi recortado al area visible): que zonas
   le quedan libres al rival cuando ataca al equipo analizado.
3. El fantasma de Jardine (ghosting por vecino mas cercano): para cada jugada
   del America de Almada se busca la situacion mas parecida del America de
   Jardine y se compara donde estaban parados los defensores.

Solo guarda resultados agregados y algunos ejemplos para dibujar
(no guarda los datos crudos de StatsBomb).

Entradas (en data/): partidos.csv, xt_grid.npy, events/, frames/
Salida:              data/ideas_extra.json

Requisitos: numpy, pandas, scipy. Opcional: POT (pip install pot) para que el
transporte optimo sea mas rapido; si no esta, usa scipy.

Uso (en la misma carpeta que construir_huella.py):
    python ideas_extra.py
    python ideas_extra.py --a Jardine-America --b Almada-America
"""
import argparse
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as Poligono
from scipy.optimize import linear_sum_assignment, linprog
from scipy.spatial import cKDTree

from construir_huella import col, leer, preparar

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(2026)

CON_BALON = {"Pass", "Carry", "Ball Receipt*", "Shot", "Dribble"}
SIN_BALON = {"Pressure", "Duel", "Interception", "Ball Recovery", "Block", "Clearance", "Foul Committed"}
PATRON_ABIERTO = {"Regular Play", "From Counter", "From Goal Kick", "From Keeper", "From Kick Off"}

# Transporte optimo: rejilla de 12 x 8 celdas de 10 yardas
OT_NX, OT_NY = 12, 8
N_REPS_RUIDO = 200

# Espacio: rejilla de 3 yardas
ESP_PASO = 3.0
ESP_MAX_FRAMES = 250          # frames muestreados por partido
ESP_MIN_OBS = 40              # observaciones minimas para reportar una celda

# Fantasma
FAN_MAX_DB = 400              # frames por partido en la base del fantasma
FAN_MAX_CONSULTA = 300        # frames por partido a comparar
FAN_RADIO_BALON = 6.0         # yardas: el balon debe estar en el mismo lugar
FAN_CANDIDATOS = 300
FAN_EJEMPLOS = 10


def r(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return round(float(v), d)


# ---------------------------------------------------------------- frames 360
def frames_por_evento(fr):
    """Devuelve dict id -> (A rivales del actor incl. actor, D defensores sin portero, K portero,
    balon, poligono visible). Coordenadas en la direccion de ataque del actor."""
    if fr is None or fr.empty:
        return {}
    # Algunas filas traen la ubicacion incompleta o vacia: se descartan
    def valido(v):
        try:
            return np.ravel(np.asarray(v, dtype=float)).size >= 2
        except (TypeError, ValueError):
            return False
    ok = fr["location"].apply(valido)
    fr = fr[ok].sort_values("id")
    if fr.empty:
        return {}
    loc = np.array([np.ravel(np.asarray(v, dtype=float))[:2] for v in fr["location"].values])
    fin = np.isfinite(loc).all(axis=1)
    fr, loc = fr[fin], loc[fin]
    if fr.empty:
        return {}
    if "visible_area" not in fr.columns:
        fr = fr.assign(visible_area=None)
    tm = fr["teammate"].values.astype(bool)
    ac = fr["actor"].values.astype(bool)
    kp = fr["keeper"].values.astype(bool)
    ids, ini = np.unique(fr["id"].values, return_index=True)
    fin = list(ini[1:]) + [len(fr)]
    va = fr["visible_area"].values
    out = {}
    for i, a, b in zip(ids, ini, fin):
        s = slice(a, b)
        act = loc[s][ac[s]]
        if len(act) == 0:
            continue
        out[i] = {
            "A": loc[s][tm[s]],
            "D": loc[s][(~tm[s]) & (~kp[s])],
            "K": loc[s][(~tm[s]) & kp[s]],
            "bal": act[0],
            "va": va[a],
        }
    return out


def poligono(va):
    v = np.asarray(va, dtype=float) if va is not None else np.array([])
    if v.size < 6:
        return None
    return Poligono(v.reshape(-1, 2))


# ----------------------------------------------------- 1. transporte optimo
CENTROS = np.array([[(i + 0.5) * 120 / OT_NX, (j + 0.5) * 80 / OT_NY]
                    for i in range(OT_NX) for j in range(OT_NY)])
COSTO = np.hypot(CENTROS[:, None, 0] - CENTROS[None, :, 0], CENTROS[:, None, 1] - CENTROS[None, :, 1])

try:
    import ot as _pot
except ImportError:
    _pot = None


def histograma(x, y):
    ix = np.clip((np.asarray(x) / 120 * OT_NX).astype(int), 0, OT_NX - 1)
    iy = np.clip((np.asarray(y) / 80 * OT_NY).astype(int), 0, OT_NY - 1)
    h = np.zeros(OT_NX * OT_NY)
    np.add.at(h, ix * OT_NY + iy, 1)
    return h


def norm(h):
    s = h.sum()
    return h / s if s > 0 else h


def emd(a, b, plan=False):
    """Distancia de Wasserstein-1 (yardas) entre dos histogramas normalizados."""
    a, b = norm(a), norm(b)
    if _pot is not None:
        G = _pot.emd(a, b, COSTO)
        w = float((G * COSTO).sum())
        return (w, G) if plan else w
    n = len(a)
    A_eq = np.zeros((2 * n, n * n))
    for i in range(n):
        A_eq[i, i * n:(i + 1) * n] = 1
        A_eq[n + i, i::n] = 1
    res = linprog(COSTO.ravel(), A_eq=A_eq[:-1], b_eq=np.concatenate([a, b])[:-1],
                  bounds=(0, None), method="highs")
    G = res.x.reshape(n, n)
    w = float(res.fun)
    return (w, G) if plan else w


def flechas(G, top=24, minimo=0.002):
    """Movimientos mas grandes del plan de transporte (fuera de la diagonal)."""
    G = G.copy()
    np.fill_diagonal(G, 0)
    idx = np.argsort(G.ravel())[::-1][:top]
    out = []
    for k in idx:
        i, j = divmod(k, G.shape[1])
        if G[i, j] < minimo:
            break
        out.append([r(CENTROS[i, 0], 1), r(CENTROS[i, 1], 1), r(CENTROS[j, 0], 1), r(CENTROS[j, 1], 1), r(G[i, j], 4)])
    return out


def desplazamiento_medio(G):
    d = CENTROS[None, :, :] - CENTROS[:, None, :]
    return [r((G * d[:, :, 0]).sum(), 2), r((G * d[:, :, 1]).sum(), 2)]


# ------------------------------------------------- 2. control del espacio
GX = int(120 / ESP_PASO)
GY = int(80 / ESP_PASO) + 1
PX, PY = np.meshgrid((np.arange(GX) + 0.5) * ESP_PASO, (np.arange(GY) + 0.5) * 80 / GY, indexing="ij")
PUNTOS = np.column_stack([PX.ravel(), PY.ravel()])


def control_frame(f):
    """Para un frame, devuelve (mascara visible, mascara rival controla) sobre PUNTOS."""
    pol = poligono(f["va"])
    if pol is None:
        return None
    vis = pol.contains_points(PUNTOS)
    A = f["A"]
    D = np.vstack([f["D"], f["K"]]) if len(f["K"]) else f["D"]
    if len(A) < 2 or len(D) < 5 or vis.sum() == 0:
        return None
    dA = np.min(np.hypot(PUNTOS[:, None, 0] - A[None, :, 0], PUNTOS[:, None, 1] - A[None, :, 1]), axis=1)
    dD = np.min(np.hypot(PUNTOS[:, None, 0] - D[None, :, 0], PUNTOS[:, None, 1] - D[None, :, 1]), axis=1)
    return vis, vis & (dA < dD)


def carril(y):
    return "izq" if y < 80 / 3 else ("centro" if y < 160 / 3 else "der")


CARRILES = {c: np.array([carril(y) == c for y in PUNTOS[:, 1]]) for c in ("izq", "centro", "der")}


# ---------------------------------------------------------- 3. fantasma
def chamfer(P, Q):
    if len(P) == 0 or len(Q) == 0:
        return 99.0
    d = np.hypot(P[:, None, 0] - Q[None, :, 0], P[:, None, 1] - Q[None, :, 1])
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def forma(D, bal):
    """Metricas del bloque en la perspectiva del que defiende (x_o = 120 - x del rival)."""
    xo = 120 - D[:, 0]
    linea = np.sort(xo)[:4].mean()
    return {
        "linea": linea,
        "altura": xo.mean(),
        "largo": xo.max() - xo.min(),
        "ancho": D[:, 1].max() - D[:, 1].min(),
        "al_balon": np.hypot(D[:, 0] - bal[0], D[:, 1] - bal[1]).min(),
        "cerca5": int((np.hypot(D[:, 0] - bal[0], D[:, 1] - bal[1]) <= 5).sum()),
    }


class BaseFantasma:
    def __init__(self):
        self.bal, self.A, self.D, self.meta = [], [], [], []

    def agregar(self, f, meta):
        self.bal.append(f["bal"])
        self.A.append(f["A"][~np.all(np.isclose(f["A"], f["bal"]), axis=1)] if len(f["A"]) else f["A"])
        self.D.append(f["D"])
        self.meta.append(meta)

    def cerrar(self):
        self.arbol = cKDTree(np.array(self.bal)) if self.bal else None
        self.partido = np.array([m["mid"] for m in self.meta]) if self.meta else np.array([])

    def buscar(self, f, excluir_partido=None):
        if self.arbol is None:
            return None
        k = min(FAN_CANDIDATOS, len(self.bal))
        dist, idx = self.arbol.query(f["bal"], k=k, distance_upper_bound=FAN_RADIO_BALON)
        dist, idx = np.atleast_1d(dist), np.atleast_1d(idx)
        ok = np.isfinite(dist)
        A = f["A"][~np.all(np.isclose(f["A"], f["bal"]), axis=1)] if len(f["A"]) else f["A"]
        mejor, mejor_d = None, np.inf
        for d0, i in zip(dist[ok], idx[ok]):
            if excluir_partido is not None and self.partido[i] == excluir_partido:
                continue
            if len(self.D[i]) < 5:
                continue
            d = chamfer(A, self.A[i]) + 0.5 * d0 + 1.5 * abs(len(A) - len(self.A[i]))
            if d < mejor_d:
                mejor, mejor_d = i, d
        return (mejor, mejor_d) if mejor is not None else None


def comparar(f, base, i):
    """Diferencias del bloque real vs el fantasma (positivo = el real esta mas adelantado)."""
    D, G = f["D"], base.D[i]
    real, fan = forma(D, f["bal"]), forma(G, f["bal"])
    c = np.hypot(D[:, None, 0] - G[None, :, 0], D[:, None, 1] - G[None, :, 1])
    fi, co = linear_sum_assignment(c)
    return {
        "linea": real["linea"] - fan["linea"],
        "altura": real["altura"] - fan["altura"],
        "largo": real["largo"] - fan["largo"],
        "ancho": real["ancho"] - fan["ancho"],
        "al_balon": real["al_balon"] - fan["al_balon"],
        "cerca5": real["cerca5"] - fan["cerca5"],
        "mov": float(c[fi, co].mean()),
    }


def resumir(difs):
    if not difs:
        return None
    df = pd.DataFrame(difs)
    out = {"n": int(len(df))}
    for k in ["linea", "altura", "largo", "ancho", "al_balon", "cerca5", "mov"]:
        v = df[k].values
        b = [RNG.choice(v, len(v)).mean() for _ in range(500)]
        out[k] = [r(v.mean(), 2), r(np.percentile(b, 5), 2), r(np.percentile(b, 95), 2)]
    return out


def zona_balon(x):
    # x en la direccion del rival: <40 su tercio, 40-80 medio, >=80 nuestro tercio
    return "Salida rival" if x < 40 else ("Medio" if x < 80 else "Nuestro tercio")


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--a", default="Jardine-America", help="caso de referencia (el fantasma)")
    ap.add_argument("--b", default="Almada-America", help="caso a comparar")
    args = ap.parse_args()
    data = Path(args.data)
    partidos = pd.read_csv(data / "partidos.csv").sort_values("match_date")
    xt = np.load(data / "xt_grid.npy") if (data / "xt_grid.npy").exists() else None
    casos = list(dict.fromkeys(partidos["caso"]))
    if args.a not in casos or args.b not in casos:
        raise SystemExit(f"No encuentro los casos {args.a} / {args.b}. Casos disponibles: {casos}")
    print(f"Motor de transporte optimo: {'POT' if _pot else 'scipy (instala pot para ir mas rapido)'}")

    h_on, h_off = defaultdict(lambda: np.zeros(OT_NX * OT_NY)), defaultdict(lambda: np.zeros(OT_NX * OT_NY))
    h_partido = {}                       # mid -> (caso, fecha, rival, h_on, h_off)
    vis_c = defaultdict(lambda: np.zeros(len(PUNTOS)))
    riv_c = defaultdict(lambda: np.zeros(len(PUNTOS)))
    esp_frames = defaultdict(int)
    esp_peligro = defaultdict(list)
    esp_carril = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))
    base = BaseFantasma()
    consultas = []                       # (mid, f, meta) del caso b
    control_a = []                       # frames del caso a para medir el ruido del fantasma

    if xt is not None:
        ix = np.clip((PUNTOS[:, 0] / 120 * xt.shape[0]).astype(int), 0, xt.shape[0] - 1)
        iy = np.clip((PUNTOS[:, 1] / 80 * xt.shape[1]).astype(int), 0, xt.shape[1] - 1)
        xt_p = xt[ix, iy]
    else:
        xt_p = np.ones(len(PUNTOS))

    t0 = time.time()
    filas = list(partidos.itertuples())
    for k, rw in enumerate(filas, 1):
        mid, caso, equipo = int(rw.match_id), rw.caso, rw.equipo
        ev = leer(data / "events" / str(mid))
        if ev is None or ev.empty:
            continue
        ev = preparar(ev)
        q = ev[(ev["team"] == equipo) & ev["x"].notna()]
        a_on, a_off = q[q["type"].isin(CON_BALON)], q[q["type"].isin(SIN_BALON)]
        h1, h2 = histograma(a_on["x"], a_on["y"]), histograma(a_off["x"], a_off["y"])
        h_on[caso] += h1
        h_off[caso] += h2
        h_partido[(mid, equipo)] = (caso, str(rw.match_date), rw.rival, h1, h2)

        # Frames 360 de jugadas del rival en juego abierto
        fr = leer(data / "frames" / str(mid))
        if fr is None or fr.empty:
            continue
        rv = ev[(ev["team"] != equipo) & ev["type"].isin(["Pass", "Carry"])
                & col(ev, "play_pattern").isin(PATRON_ABIERTO)]
        fr = fr[fr["id"].isin(rv["id"])]
        frames = frames_por_evento(fr)
        if not frames:
            continue
        ids = np.array(list(frames.keys()))

        # 2. espacio (muestra)
        mues = RNG.choice(ids, min(ESP_MAX_FRAMES, len(ids)), replace=False)
        for i in mues:
            f = frames[i]
            if f["bal"][0] < 40:
                continue
            res = control_frame(f)
            if res is None:
                continue
            vis, ctl = res
            vis_c[caso] += vis
            riv_c[caso] += ctl
            esp_frames[caso] += 1
            den = (xt_p * vis).sum()
            if den > 0:
                esp_peligro[caso].append((xt_p * ctl).sum() / den)
            ult = vis & (PUNTOS[:, 0] >= 80)
            for c_, mc in CARRILES.items():
                m = ult & mc
                esp_carril[caso][c_] += [ctl[m].sum(), m.sum()]

        # 3. fantasma
        if caso in (args.a, args.b):
            info = rv.set_index("id")
            asist = set(rv.loc[col(rv, "pass_shot_assist").fillna(False).astype(bool)
                               | col(rv, "pass_goal_assist").fillna(False).astype(bool), "id"])
            tiros = ev[ev["type"] == "Shot"].set_index("id")
            if caso == args.a:
                for i in RNG.choice(ids, min(FAN_MAX_DB, len(ids)), replace=False):
                    if len(frames[i]["D"]) >= 5:
                        base.agregar(frames[i], {"mid": mid, "fecha": str(rw.match_date), "rival": rw.rival})
                for i in RNG.choice(ids, min(40, len(ids)), replace=False):
                    if len(frames[i]["D"]) >= 5:
                        control_a.append((mid, frames[i]))
            else:
                elegidos = set(RNG.choice(ids, min(FAN_MAX_CONSULTA, len(ids)), replace=False)) | (asist & set(ids))
                for i in elegidos:
                    f = frames[i]
                    if len(f["D"]) < 5:
                        continue
                    meta = {"mid": mid, "fecha": str(rw.match_date), "rival": rw.rival, "asist": i in asist}
                    if i in asist:
                        sid = info.loc[i].get("pass_assisted_shot_id") if "pass_assisted_shot_id" in info.columns else None
                        if isinstance(sid, str) and sid in tiros.index:
                            t = tiros.loc[sid]
                            meta["xg"] = r(t.get("shot_statsbomb_xg", 0) or 0, 3)
                            meta["gol"] = bool(t.get("shot_outcome") == "Goal")
                            meta["min"] = int(t.get("minute", 0))
                    consultas.append((mid, f, meta))
        if k % 25 == 0 or k == len(filas):
            print(f"  {k}/{len(filas)} partidos ({time.time() - t0:.0f} s)")

    salida = {"casos": casos, "a": args.a, "b": args.b}

    # ---------------- 1. transporte optimo
    print("\n1. Transporte optimo")
    tr = {"rejilla": [OT_NX, OT_NY], "pares": {}, "ruido": {}}
    for nombre, H in (("con_balon", h_on), ("sin_balon", h_off)):
        w, G = emd(H[args.a], H[args.b], plan=True)
        diff = (norm(H[args.b]) - norm(H[args.a])).reshape(OT_NX, OT_NY)
        # Ruido: misma cantidad de partidos, pero todos del caso a
        mids_a = [m for m, v in h_partido.items() if v[0] == args.a]
        n_b = sum(1 for v in h_partido.values() if v[0] == args.b)
        ruido = []
        j = 3 if nombre == "con_balon" else 4
        for _ in range(N_REPS_RUIDO if len(mids_a) > n_b else 0):
            sub = RNG.choice(len(mids_a), n_b, replace=False)
            ruido.append(emd(H[args.a], sum(h_partido[mids_a[s]][j] for s in sub)))
        tr[nombre] = {
            "w": r(w, 2),
            "ruido_media": r(np.mean(ruido), 2) if ruido else None,
            "ruido_p95": r(np.percentile(ruido, 95), 2) if ruido else None,
            "p_valor": r((np.sum(np.array(ruido) >= w) + 1) / (len(ruido) + 1), 4) if ruido else None,
            "flechas": flechas(G),
            "desplazamiento": desplazamiento_medio(G),
            "a": [r(v, 4) for v in norm(H[args.a])],
            "b": [r(v, 4) for v in norm(H[args.b])],
            "diferencia": [[r(v, 4) for v in fila] for fila in diff],
        }
        tr["pares"][nombre] = {c1: {c2: r(emd(H[c1], H[c2]), 2) for c2 in casos} for c1 in casos}
        print(f"  {nombre}: W({args.a} -> {args.b}) = {w:.2f} yardas;"
              f" ruido {tr[nombre]['ruido_media']} (p95 {tr[nombre]['ruido_p95']}), p = {tr[nombre]['p_valor']}")
    # Trayectoria partido a partido del caso b
    otros = [c for c in casos if c not in (args.a, args.b) and "Almada" in c] or [c for c in casos if c not in (args.a, args.b)][:1]
    tray = []
    for m, (c, fecha, rival, h1, _) in h_partido.items():
        if c == args.b:
            fila = {"fecha": fecha, "rival": rival, "a": r(emd(h_on[args.a], h1), 2)}
            for o in otros:
                fila[o] = r(emd(h_on[o], h1), 2)
            tray.append(fila)
    claves_a = [m for m, v in h_partido.items() if v[0] == args.a]
    ruido_partido = [emd(h_on[args.a], h_partido[claves_a[i]][3])
                     for i in RNG.choice(len(claves_a), min(40, len(claves_a)), replace=False)]
    tr["trayectoria"] = sorted(tray, key=lambda d: d["fecha"])
    tr["trayectoria_ref"] = otros
    tr["ruido_partido"] = r(np.mean(ruido_partido), 2) if ruido_partido else None
    salida["transporte"] = tr

    # ---------------- 2. control del espacio
    print("\n2. Control del espacio (frames 360 con el balon del rival en su mitad de ataque o medio)")
    esp = {"paso": ESP_PASO, "gx": GX, "gy": GY, "casos": {}}
    for c in casos:
        if esp_frames[c] == 0:
            continue
        p = np.where(vis_c[c] >= ESP_MIN_OBS, riv_c[c] / np.maximum(vis_c[c], 1), np.nan).reshape(GX, GY)
        ult = p[int(80 / ESP_PASO):, :]
        car = {k: r(v[0] / v[1], 3) if v[1] else None for k, v in esp_carril[c].items()}
        esp["casos"][c] = {
            "frames": int(esp_frames[c]),
            "rejilla": [[r(v, 3) for v in fila] for fila in p],
            "peligro_libre": r(np.mean(esp_peligro[c]), 4) if esp_peligro[c] else None,
            "ultimo_tercio": r(np.nanmean(ult), 4) if np.isfinite(ult).any() else None,
            "carril_ultimo_tercio": car,
        }
        print(f"  {c:<22} frames {esp_frames[c]:>6}  espacio peligroso del rival {esp['casos'][c]['peligro_libre']}"
              f"  ultimo tercio {esp['casos'][c]['ultimo_tercio']}  carriles {car}")
    salida["espacio"] = esp

    # ---------------- 3. fantasma
    print(f"\n3. Fantasma de {args.a} sobre {args.b}")
    base.cerrar()
    difs, difs_zona, ejemplos = [], defaultdict(list), []
    for mid, f, meta in consultas:
        hit = base.buscar(f)
        if hit is None:
            continue
        i, d = hit
        dif = comparar(f, base, i)
        difs.append(dif)
        difs_zona[zona_balon(f["bal"][0])].append(dif)
        if meta.get("asist"):
            ejemplos.append((meta, f, i, d))
    ruido = []
    for mid, f in control_a[:1500]:
        hit = base.buscar(f, excluir_partido=mid)
        if hit is not None:
            ruido.append(comparar(f, base, hit[0]))
    ejemplos.sort(key=lambda e: (not e[0].get("gol", False), -(e[0].get("xg") or 0), e[3]))
    fan = {
        "global": resumir(difs),
        "por_zona": {z: resumir(v) for z, v in difs_zona.items() if len(v) >= 20},
        "ruido": resumir(ruido),
        "ejemplos": [],
    }
    for meta, f, i, d in ejemplos[:FAN_EJEMPLOS]:
        g = base.meta[i]
        fan["ejemplos"].append({
            "rival": meta["rival"], "fecha": meta["fecha"], "min": meta.get("min"),
            "xg": meta.get("xg"), "gol": meta.get("gol", False),
            "fantasma_de": f"{g['rival']} ({g['fecha'][:4]})",
            "similitud": r(d, 2),
            "balon": [r(v, 1) for v in f["bal"]],
            "atacantes": [[r(x, 1), r(y, 1)] for x, y in f["A"]],
            "real": [[r(x, 1), r(y, 1)] for x, y in f["D"]],
            "fantasma": [[r(x, 1), r(y, 1)] for x, y in base.D[i]],
            "visible": [r(v, 1) for v in np.asarray(f["va"], dtype=float)],
            "dif": {k: r(v, 2) for k, v in comparar(f, base, i).items()},
        })
    salida["fantasma"] = fan
    g = fan["global"]
    if g:
        print(f"  Situaciones comparadas: {g['n']}  (ruido medido con {fan['ruido']['n'] if fan['ruido'] else 0})")
        for k, t in [("linea", "linea defensiva (+ = mas arriba)"), ("altura", "altura media del bloque"),
                     ("largo", "largo del bloque"), ("ancho", "ancho del bloque"),
                     ("al_balon", "distancia del defensor mas cercano al balon"),
                     ("cerca5", "defensores a 5 yd del balon"), ("mov", "movimiento medio por jugador")]:
            rr = fan["ruido"][k][0] if fan["ruido"] else None
            print(f"    {t:<46} {g[k][0]:>6} [{g[k][1]}, {g[k][2]}]   ruido {rr}")
        for z, v in fan["por_zona"].items():
            if v:
                print(f"    {z:<16} linea {v['linea'][0]:>6}  al balon {v['al_balon'][0]:>6}  n {v['n']}")
        print(f"  Ejemplos guardados (pases que terminaron en tiro rival): {len(fan['ejemplos'])}")

    ruta = data / "ideas_extra.json"
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(salida, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"\nGuardado: {ruta} ({ruta.stat().st_size / 1e6:.2f} MB) en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
