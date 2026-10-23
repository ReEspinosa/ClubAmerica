"""
Paso 2 - Huella de estilo (Hackathon ISAC 2026).

Calcula, para CADA partido y CADA equipo (casos de estudio y rivales),
el vector de indicadores definido en el documento "Framework analitico".
Los rivales forman la referencia de "Liga MX promedio".

Entradas (generadas por descargar_casos.py):
    data/partidos.csv
    data/events/<match_id>.parquet  (o .pkl)

Salidas:
    data/huella.csv     una fila por (partido, equipo)
    data/xt_grid.npy    rejilla de Expected Threat (16 x 12) estimada con todos los partidos

Uso:
    python construir_huella.py
    python construir_huella.py --data otra_carpeta
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Parametros del framework (si cambian aqui, cambian en todo el analisis)
# ---------------------------------------------------------------------------
LARGO_PASE = 35.0          # yardas (aprox. 32 m): umbral de pase largo
TERCIO_FINAL = 80.0        # x >= 80 es el ultimo tercio
TERCIO_PROPIO = 40.0       # x < 40 es el tercio propio
ZONA_PPDA_RIVAL = 72.0     # 60% defensivo del rival: x_rival < 72
ZONA_PPDA_PROPIA = 48.0    # misma zona vista desde el equipo que defiende: x >= 48
VENTANA_TRANSICION = 10.0  # segundos
PROGRESIVO = 0.75          # termina a <= 75% de la distancia inicial a porteria
XT_NX, XT_NY = 16, 12      # rejilla de xT (Markov)
PRIOR_TIROS = 10           # suavizado de la probabilidad de gol por zona
ENT_NX, ENT_NY = 6, 4      # rejilla gruesa para entropias
PORTERIA = np.array([120.0, 40.0])

TIPOS_RECUPERACION = {"Ball Recovery", "Interception", "Duel", "Block", "Clearance"}
TIPOS_PPDA = {"Interception", "Foul Committed"}  # + Duel de tipo Tackle (abajo)
PASES_JUEGO_ABIERTO = {np.nan, None, "Recovery", "Interception"}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def col(df, nombre, default=np.nan):
    return df[nombre] if nombre in df.columns else pd.Series(default, index=df.index)


def xy(serie):
    """Convierte una columna de listas [x, y] en dos Series numericas."""
    def g(v, i):
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) > i:
            return float(v[i])
        return np.nan
    return serie.apply(lambda v: g(v, 0)), serie.apply(lambda v: g(v, 1))


def leer(ruta_sin_ext):
    p = ruta_sin_ext.with_suffix(".parquet")
    if p.exists():
        return pd.read_parquet(p)
    p = ruta_sin_ext.with_suffix(".pkl")
    if p.exists():
        return pd.read_pickle(p)
    return None


def preparar(ev):
    """Agrega columnas numericas y de tiempo a un DataFrame de eventos."""
    ev = ev.copy()
    ev = ev[col(ev, "period", 1).fillna(1) <= 4]  # sin tanda de penales
    ev["x"], ev["y"] = xy(col(ev, "location", None))
    ev["pex"], ev["pey"] = xy(col(ev, "pass_end_location", None))
    ev["cex"], ev["cey"] = xy(col(ev, "carry_end_location", None))
    ev["t"] = col(ev, "minute", 0).fillna(0) * 60 + col(ev, "second", 0).fillna(0)

    # Tiempo absoluto: cada periodo empieza donde termino el anterior
    offs, acumulado = {}, 0.0
    for p, g in ev.groupby("period"):
        offs[p] = acumulado - g["t"].min()
        acumulado += g["t"].max() - g["t"].min()
    ev["t_abs"] = ev["t"] + ev["period"].map(offs)
    ev.attrs["duracion"] = max(acumulado, 1.0)
    return ev.sort_values(["t_abs", "index"] if "index" in ev.columns else "t_abs")


def es_pase_completo(ev):
    return (ev["type"] == "Pass") & col(ev, "pass_outcome").isna()


def es_juego_abierto(ev):
    return col(ev, "pass_type").isna() | col(ev, "pass_type").isin(["Recovery", "Interception"])


def celda(x, y, nx, ny):
    ix = np.clip((x / 120.0 * nx).astype("float"), 0, nx - 1e-9)
    iy = np.clip((y / 80.0 * ny).astype("float"), 0, ny - 1e-9)
    return np.floor(ix).astype(int), np.floor(iy).astype(int)


def entropia(conteos):
    c = np.asarray(conteos, dtype=float)
    c = c[c > 0]
    if c.sum() == 0:
        return np.nan
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------------------
# Expected Threat (Karun Singh) como cadena de Markov
# ---------------------------------------------------------------------------
def estimar_xt(eventos, iteraciones=60):
    """xT_i = P(tiro|i) P(gol|tiro,i) + P(mover|i) * sum_j T_ij xT_j"""
    n = XT_NX * XT_NY
    tiros = np.zeros(n); goles = np.zeros(n); movs = np.zeros(n); T = np.zeros((n, n))

    for ev in eventos:
        ok = ev["x"].notna() & ev["y"].notna()
        # Tiros (sin penales)
        sh = ev[ok & (ev["type"] == "Shot") & (col(ev, "shot_type") != "Penalty")]
        ix, iy = celda(sh["x"].values, sh["y"].values, XT_NX, XT_NY)
        k = ix * XT_NY + iy
        np.add.at(tiros, k, 1)
        np.add.at(goles, k[(col(sh, "shot_outcome") == "Goal").values], 1)

        # Movimientos: pases (todos los intentos) y conducciones
        pa = ev[ok & (ev["type"] == "Pass") & es_juego_abierto(ev)]
        ix, iy = celda(pa["x"].values, pa["y"].values, XT_NX, XT_NY)
        k0 = ix * XT_NY + iy
        np.add.at(movs, k0, 1)
        comp = pa[col(pa, "pass_outcome").isna() & pa["pex"].notna()]
        ix, iy = celda(comp["x"].values, comp["y"].values, XT_NX, XT_NY)
        jx, jy = celda(comp["pex"].values, comp["pey"].values, XT_NX, XT_NY)
        np.add.at(T, (ix * XT_NY + iy, jx * XT_NY + jy), 1)

        ca = ev[ok & (ev["type"] == "Carry") & ev["cex"].notna()]
        ix, iy = celda(ca["x"].values, ca["y"].values, XT_NX, XT_NY)
        jx, jy = celda(ca["cex"].values, ca["cey"].values, XT_NX, XT_NY)
        np.add.at(movs, ix * XT_NY + iy, 1)
        np.add.at(T, (ix * XT_NY + iy, jx * XT_NY + jy), 1)

    total = tiros + movs
    p_tiro = np.divide(tiros, total, out=np.zeros(n), where=total > 0)
    p_mov = np.divide(movs, total, out=np.zeros(n), where=total > 0)
    # Suavizado bayesiano: zonas con pocos tiros se jalan a la conversion global
    g0 = goles.sum() / max(tiros.sum(), 1)
    p_gol = (goles + PRIOR_TIROS * g0) / (tiros + PRIOR_TIROS)
    T = np.divide(T, movs[:, None], out=np.zeros_like(T), where=movs[:, None] > 0)

    xt = np.zeros(n)
    for _ in range(iteraciones):
        xt = p_tiro * p_gol + p_mov * (T @ xt)
    return xt.reshape(XT_NX, XT_NY)


# ---------------------------------------------------------------------------
# Indicadores por partido y equipo
# ---------------------------------------------------------------------------
def minutos_por_estado(ev, equipo):
    """Minutos del partido que el equipo paso ganando, empatando y perdiendo."""
    goles = ev[((ev["type"] == "Shot") & (col(ev, "shot_outcome") == "Goal"))
               | (ev["type"] == "Own Goal For")][["t_abs", "team"]]
    dur = ev.attrs["duracion"]
    t_prev, dif = 0.0, 0
    out = {"min_ganando": 0.0, "min_empatando": 0.0, "min_perdiendo": 0.0}
    for _, g in goles.iterrows():
        clave = "min_ganando" if dif > 0 else "min_perdiendo" if dif < 0 else "min_empatando"
        out[clave] += (g["t_abs"] - t_prev) / 60
        dif += 1 if g["team"] == equipo else -1
        t_prev = g["t_abs"]
    clave = "min_ganando" if dif > 0 else "min_perdiendo" if dif < 0 else "min_empatando"
    out[clave] += (dur - t_prev) / 60
    return out


def huella_equipo(ev, equipo, xt):
    f = {}
    dur = ev.attrs["duracion"]
    p90 = 90 * 60 / dur
    eq = ev[ev["team"] == equipo]
    rv = ev[ev["team"] != equipo]
    rival = rv["team"].iloc[0] if len(rv) else None
    f["minutos"] = round(dur / 60, 1)

    pases = eq[eq["type"] == "Pass"]
    pases_ab = pases[es_juego_abierto(pases)]
    comp_ab = pases_ab[col(pases_ab, "pass_outcome").isna()]
    largo = col(pases, "pass_length")

    # --- Construccion ---
    gk = pases[col(pases, "position").astype(str).str.contains("Goalkeeper", na=False)]
    f["salida_corta_portero"] = (col(gk, "pass_length") < LARGO_PASE).mean() if len(gk) else np.nan
    f["pases_largos_pct"] = (col(pases_ab, "pass_length") >= LARGO_PASE).mean() if len(pases_ab) else np.nan
    comp_eq = es_pase_completo(eq).sum()
    comp_tot = es_pase_completo(ev).sum()
    f["posesion_pct"] = comp_eq / comp_tot if comp_tot else np.nan

    # --- Progresion ---
    def progresivos(df, ex, ey):
        d0 = np.hypot(PORTERIA[0] - df["x"], PORTERIA[1] - df["y"])
        d1 = np.hypot(PORTERIA[0] - df[ex], PORTERIA[1] - df[ey])
        return (d1 <= PROGRESIVO * d0) & df[ex].notna()
    f["pases_progresivos_p90"] = progresivos(comp_ab, "pex", "pey").sum() * p90
    carr = eq[(eq["type"] == "Carry") & eq["cex"].notna()]
    f["conducciones_progresivas_p90"] = progresivos(carr, "cex", "cey").sum() * p90

    movs = pd.concat([
        comp_ab.assign(ex=comp_ab["pex"], ey=comp_ab["pey"]),
        carr.assign(ex=carr["cex"], ey=carr["cey"]),
    ])
    movs = movs[movs["x"].notna() & movs["ex"].notna()]
    if len(movs):
        ix, iy = celda(movs["x"].values, movs["y"].values, XT_NX, XT_NY)
        jx, jy = celda(movs["ex"].values, movs["ey"].values, XT_NX, XT_NY)
        movs["dxt"] = xt[jx, jy] - xt[ix, iy]
        f["xt_por_posesion"] = movs.groupby("possession")["dxt"].sum().mean()
        f["xt_p90"] = movs["dxt"].clip(lower=0).sum() * p90
    else:
        f["xt_por_posesion"] = f["xt_p90"] = np.nan

    entradas = movs[(movs["x"] < TERCIO_FINAL) & (movs["ex"] >= TERCIO_FINAL)]
    n_ent = len(entradas)
    f["entradas_ultimo_tercio_p90"] = n_ent * p90
    for nombre, lo, hi in [("izq", 0, 80 / 3), ("centro", 80 / 3, 160 / 3), ("der", 160 / 3, 81)]:
        f[f"carril_{nombre}_pct"] = ((entradas["ey"] >= lo) & (entradas["ey"] < hi)).mean() if n_ent else np.nan

    # --- Ocasiones ---
    def npxg(df):
        s = df[(df["type"] == "Shot") & (col(df, "shot_type") != "Penalty")]
        return s, col(s, "shot_statsbomb_xg").fillna(0)
    tiros, xg = npxg(eq)
    f["tiros_p90"] = len(tiros) * p90
    f["npxg_p90"] = xg.sum() * p90
    f["xg_por_tiro"] = xg.mean() if len(tiros) else np.nan

    # Transiciones: posesion propia que empieza con recuperacion en juego abierto
    inicio = eq.groupby("possession").first()
    patron_ok = col(inicio, "play_pattern").isin(["Regular Play", "From Counter"])
    rec = inicio[patron_ok & inicio["type"].isin(TIPOS_RECUPERACION)]
    if len(tiros):
        t_ini = tiros["possession"].map(rec["t_abs"])
        en_trans = (tiros["t_abs"] - t_ini) <= VENTANA_TRANSICION
        f["tiros_transicion_pct"] = en_trans.fillna(False).mean()
    else:
        f["tiros_transicion_pct"] = np.nan

    # --- Presion ---
    pases_rival_zona = rv[(rv["type"] == "Pass") & (rv["x"] < ZONA_PPDA_RIVAL)]
    defensa = eq[(eq["x"] >= ZONA_PPDA_PROPIA) & (
        eq["type"].isin(TIPOS_PPDA)
        | ((eq["type"] == "Duel") & (col(eq, "duel_type") == "Tackle")))]
    f["ppda"] = len(pases_rival_zona) / len(defensa) if len(defensa) else np.nan
    pres = eq[eq["type"] == "Pressure"]
    f["presiones_p90"] = len(pres) * p90
    f["presion_alta_pct"] = (pres["x"] >= TERCIO_FINAL).mean() if len(pres) else np.nan
    n_pos = eq["possession"].nunique()
    f["contrapresion_por_posesion"] = (col(pres, "counterpress") == True).sum() / n_pos if n_pos else np.nan

    # --- Defensa ---
    recs = eq[eq["type"].isin(["Ball Recovery", "Interception"])
              & (col(eq, "ball_recovery_recovery_failure") != True)]
    f["altura_recuperacion"] = recs["x"].mean()
    _, xg_c = npxg(rv)
    f["npxg_contra_p90"] = xg_c.sum() * p90

    # --- Dominio ---
    pt_eq = ((eq["type"] == "Pass") & (eq["x"] >= TERCIO_FINAL)).sum()
    pt_rv = ((rv["type"] == "Pass") & (rv["x"] >= TERCIO_FINAL)).sum()
    f["field_tilt"] = pt_eq / (pt_eq + pt_rv) if (pt_eq + pt_rv) else np.nan

    # --- Predictibilidad ---
    pos = col(comp_ab, "position").astype(str)
    salida = comp_ab[pos.str.contains("Goalkeeper|Center Back", na=False) & comp_ab["pex"].notna()]
    if len(salida) >= 5:
        jx, jy = celda(salida["pex"].values, salida["pey"].values, ENT_NX, ENT_NY)
        f["entropia_salida"] = entropia(np.bincount(jx * ENT_NY + jy, minlength=ENT_NX * ENT_NY))
    else:
        f["entropia_salida"] = np.nan

    if len(movs) >= 20:
        ix, iy = celda(movs["x"].values, movs["y"].values, ENT_NX, ENT_NY)
        jx, jy = celda(movs["ex"].values, movs["ey"].values, ENT_NX, ENT_NY)
        n = ENT_NX * ENT_NY
        M = np.zeros((n, n))
        np.add.at(M, (ix * ENT_NY + iy, jx * ENT_NY + jy), 1)
        filas = M.sum(1)
        pi = filas / filas.sum()
        h = sum(pi[i] * entropia(M[i]) for i in range(n) if filas[i] > 0)
        f["entropia_progresion"] = h
    else:
        f["entropia_progresion"] = np.nan

    f.update(minutos_por_estado(ev, equipo))
    return rival, f


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    data = Path(args.data)

    partidos = pd.read_csv(data / "partidos.csv")
    casos = {(int(r.match_id), r.equipo): r for r in partidos.itertuples()}
    ids = sorted(partidos["match_id"].astype(int).unique())

    print(f"Cargando eventos de {len(ids)} partidos...")
    eventos, faltan = {}, []
    for mid in ids:
        ev = leer(data / "events" / str(mid))
        if ev is None or ev.empty:
            faltan.append(mid)
            continue
        eventos[mid] = preparar(ev)
    if faltan:
        print(f"  Sin archivo de eventos (se omiten): {faltan}")

    print("Estimando la rejilla de Expected Threat (cadena de Markov)...")
    xt = estimar_xt(eventos.values())
    np.save(data / "xt_grid.npy", xt)
    print(f"  xT maximo {xt.max():.3f}, minimo {xt.min():.4f}")

    print("Calculando la huella por partido y equipo...")
    filas = []
    for k, (mid, ev) in enumerate(eventos.items(), 1):
        for equipo in ev["team"].dropna().unique():
            rival, f = huella_equipo(ev, equipo, xt)
            meta = casos.get((mid, equipo))
            filas.append({
                "match_id": mid,
                "equipo": equipo,
                "rival": rival,
                "caso": meta.caso if meta is not None else "Liga",
                "entrenador": meta.entrenador if meta is not None else None,
                "match_date": meta.match_date if meta is not None else None,
                **f,
            })
        if k % 50 == 0:
            print(f"  {k}/{len(eventos)}")

    huella = pd.DataFrame(filas)
    # Fecha, temporada, fase y localia para todas las filas (tambien rivales)
    meta = partidos.drop_duplicates("match_id").set_index("match_id")
    huella["match_date"] = huella["match_id"].map(meta["match_date"])
    for c in ["season", "competition_stage"]:
        if c in meta.columns:
            huella[c] = huella["match_id"].map(meta[c])
    huella = huella.sort_values(["match_date", "match_id", "caso"]).reset_index(drop=True)
    huella.to_csv(data / "huella.csv", index=False)

    print(f"\nGuardado: {data / 'huella.csv'} ({len(huella)} filas, {huella.shape[1]} columnas)")
    resumen_cols = ["posesion_pct", "pases_largos_pct", "npxg_p90", "npxg_contra_p90",
                    "ppda", "presion_alta_pct", "field_tilt", "entropia_progresion"]
    print("\nPrimer vistazo (promedio por caso):")
    print(huella.groupby("caso")[resumen_cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
