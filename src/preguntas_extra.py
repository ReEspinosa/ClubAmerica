"""
Preguntas extra (Hackathon ISAC 2026):

  2. Medio tiempo: cuanto cambia el equipo del primer al segundo tiempo y si el
     entrenador "corrige" cuando no va ganando al descanso. Referencia: los rivales
     de esos mismos partidos, en la misma situacion de marcador.
  3. Desarrollo de jugadores: los mismos futbolistas del America con Ortiz, con
     Jardine y con Almada (por 90 minutos).

Solo guarda resultados agregados.

Entradas (en data/): partidos.csv, xt_grid.npy, events/, lineups/
Salida:              data/preguntas_extra.json

Uso (misma carpeta que construir_huella.py y analisis_completo.py):
    python preguntas_extra.py
"""
import argparse
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from construir_huella import col, huella_equipo, leer, preparar
from analisis_completo import minutos_jugador

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(2026)

KPI_MT = ["posesion_pct", "field_tilt", "ppda", "presion_alta_pct", "pases_progresivos_p90",
          "xt_p90", "tiros_p90", "npxg_p90", "npxg_contra_p90", "pases_largos_pct"]
NOMBRE_KPI = {"posesion_pct": "Posesion", "field_tilt": "Field tilt", "ppda": "PPDA",
              "presion_alta_pct": "Presion alta", "pases_progresivos_p90": "Pases progresivos /90",
              "xt_p90": "xT /90", "tiros_p90": "Tiros /90", "npxg_p90": "xG a favor /90",
              "npxg_contra_p90": "xG en contra /90", "pases_largos_pct": "Pases largos"}
CASOS_AME = ["Ortiz-America", "Jardine-America", "Almada-America"]
MIN_MINUTOS = {"Ortiz-America": 450, "Jardine-America": 450, "Almada-America": 270}
DEF_TIPOS = {"Interception", "Ball Recovery", "Block", "Clearance"}


def r(v, d=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, d) if np.isfinite(v) else None


def boot_media(v, n=2000):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)])
    if len(v) < 3:
        return [r(v.mean()) if len(v) else None, None, None, int(len(v))]
    b = RNG.choice(v, (n, len(v))).mean(1)
    return [r(v.mean()), r(np.percentile(b, 5)), r(np.percentile(b, 95)), int(len(v))]


def goles_periodo(ev, equipo, periodo):
    g = ev[(ev["period"] == periodo) & (((ev["type"] == "Shot") & (col(ev, "shot_outcome") == "Goal"))
                                       | (ev["type"] == "Own Goal For"))]
    return int((g["team"] == equipo).sum()), int((g["team"] != equipo).sum())


def estado_txt(a, b):
    return "Ganando" if a > b else ("Perdiendo" if a < b else "Empatando")


def stats_jugador(ev, equipo):
    """Totales por jugador (se convierten a /90 al final)."""
    q = ev[ev["team"] == equipo]
    if q.empty:
        return pd.DataFrame()
    tiros = ev[ev["type"] == "Shot"]
    xg_tiro = col(tiros, "shot_statsbomb_xg").fillna(0)
    no_pen = col(tiros, "shot_type") != "Penalty"
    pases = q[q["type"] == "Pass"]
    ok = col(pases, "pass_outcome").isna()
    sid = col(pases, "pass_assisted_shot_id")
    xa = sid.map(dict(zip(tiros["id"], xg_tiro))).fillna(0)
    d0 = np.hypot(120 - pases["x"], 40 - pases["y"])
    d1 = np.hypot(120 - pases["pex"], 40 - pases["pey"])
    prog = ok & (d1 <= 0.75 * d0) & (pases["x"] >= 40)
    car = q[q["type"] == "Carry"]
    c0 = np.hypot(120 - car["x"], 40 - car["y"])
    c1 = np.hypot(120 - car["cex"], 40 - car["cey"])
    defen = q["type"].isin(DEF_TIPOS) | ((q["type"] == "Duel") & (col(q, "duel_type") == "Tackle"))
    t = pd.DataFrame({
        "npxg": tiros[(tiros["team"] == equipo) & no_pen].groupby("player").apply(lambda g: xg_tiro.loc[g.index].sum()),
        "xa": xa.groupby(pases["player"]).sum(),
        "pases": pases.groupby("player").size(),
        "pases_ok": ok.groupby(pases["player"]).sum(),
        "pases_prog": prog.groupby(pases["player"]).sum(),
        "cond_prog": ((c1 <= 0.75 * c0) & (car["x"] >= 40)).groupby(car["player"]).sum(),
        "presiones": (q["type"] == "Pressure").groupby(q["player"]).sum(),
        "acciones_def": defen.groupby(q["player"]).sum(),
        "obv": col(q, "obv_total_net").fillna(0).groupby(q["player"]).sum() if "obv_total_net" in q.columns else np.nan,
    }).fillna(0)
    pos = q.dropna(subset=["position"]).groupby("player")["position"].agg(lambda s: s.mode().iloc[0])
    t["posicion"] = pos.reindex(t.index)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    data = Path(args.data)
    partidos = pd.read_csv(data / "partidos.csv").sort_values("match_date")
    xt = np.load(data / "xt_grid.npy")
    casos = list(dict.fromkeys(partidos["caso"]))

    filas_mt = []                     # una fila por partido y equipo (caso o rival de referencia)
    jug = defaultdict(lambda: defaultdict(float))   # (caso, jugador) -> totales
    jug_pos = defaultdict(lambda: defaultdict(float))
    t0 = time.time()
    filas = list(partidos.itertuples())
    for k, rw in enumerate(filas, 1):
        mid, caso, equipo = int(rw.match_id), rw.caso, rw.equipo
        crudo = leer(data / "events" / str(mid))
        if crudo is None or crudo.empty:
            continue
        ev = preparar(crudo)
        rivales = ev.loc[ev["team"] != equipo, "team"].dropna()
        rival = rivales.iloc[0] if len(rivales) else None

        # ---- 2. medio tiempo ----
        a1, b1 = goles_periodo(ev, equipo, 1)
        a2, b2 = goles_periodo(ev, equipo, 2)
        fin_a, fin_b = a1 + a2, b1 + b2
        mitades = {}
        for p in (1, 2):
            sub = crudo[col(crudo, "period", 1) == p]
            if len(sub) < 200:
                break
            sp = preparar(sub)
            mitades[p] = {eq: huella_equipo(sp, eq, xt)[1] for eq in (equipo, rival) if eq}
        if len(mitades) == 2:
            for eq, grupo, (h1, h2, f1, f2) in ((equipo, caso, (a1, b1, fin_a, fin_b)),
                                                (rival, "Liga", (b1, a1, fin_b, fin_a))):
                if not eq or eq not in mitades[1] or eq not in mitades[2]:
                    continue
                fila = {"caso": grupo, "match_id": mid, "estado_mt": estado_txt(h1, h2),
                        "estado_final": estado_txt(f1, f2),
                        "pts_mt": 3 if h1 > h2 else (1 if h1 == h2 else 0),
                        "pts_final": 3 if f1 > f2 else (1 if f1 == f2 else 0)}
                for kpi in KPI_MT:
                    v1, v2 = mitades[1][eq].get(kpi), mitades[2][eq].get(kpi)
                    fila[f"{kpi}_1"], fila[f"{kpi}_2"] = v1, v2
                    fila[f"d_{kpi}"] = (v2 - v1) if v1 is not None and v2 is not None else np.nan
                d1 = mitades[1][eq].get("npxg_p90", np.nan) - mitades[1][eq].get("npxg_contra_p90", np.nan)
                d2 = mitades[2][eq].get("npxg_p90", np.nan) - mitades[2][eq].get("npxg_contra_p90", np.nan)
                fila["d_xg_dif"] = d2 - d1
                filas_mt.append(fila)

        # ---- 3. jugadores ----
        if caso in CASOS_AME:
            lu = leer(data / "lineups" / str(mid))
            mins = minutos_jugador(lu, equipo, ev.attrs["duracion"] / 60)
            st = stats_jugador(ev, equipo)
            for j, m in mins.items():
                jug[(caso, j)]["minutos"] += m
                jug[(caso, j)]["partidos"] += 1
                if j in st.index:
                    for c_ in st.columns:
                        if c_ != "posicion":
                            jug[(caso, j)][c_] += float(st.loc[j, c_])
                    if isinstance(st.loc[j, "posicion"], str):
                        jug_pos[j][st.loc[j, "posicion"]] += m
        if k % 25 == 0 or k == len(filas):
            print(f"  {k}/{len(filas)} partidos ({time.time() - t0:.0f} s)")

    salida = {}

    # ================= 2. MEDIO TIEMPO =================
    mt = pd.DataFrame(filas_mt)
    print("\n2. Medio tiempo: cambio del 1T al 2T (2T - 1T)")
    res_mt = {}
    for g in casos + ["Liga"]:
        s = mt[mt["caso"] == g]
        if s.empty:
            continue
        out = {"partidos": int(len(s)), "kpis": {}, "por_estado": {}, "remontadas": {}}
        for kpi in KPI_MT:
            out["kpis"][kpi] = {"t1": r(s[f"{kpi}_1"].mean()), "t2": r(s[f"{kpi}_2"].mean()),
                                "cambio": boot_media(s[f"d_{kpi}"].values)}
        for est in ["Perdiendo", "Empatando", "Ganando"]:
            e = s[s["estado_mt"] == est]
            if len(e) == 0:
                continue
            out["por_estado"][est] = {
                "partidos": int(len(e)),
                "d_xg_dif": boot_media(e["d_xg_dif"].values),
                "d_posesion": r(e["d_posesion_pct"].mean()), "d_ppda": r(e["d_ppda"].mean()),
                "d_field_tilt": r(e["d_field_tilt"].mean()),
                "pts_mt": r(e["pts_mt"].mean(), 2), "pts_final": r(e["pts_final"].mean(), 2),
                "pts_ganados_2t": r((e["pts_final"] - e["pts_mt"]).mean(), 2),
            }
        out["remontadas"] = {
            "perdiendo_y_no_perdio": r((s.loc[s["estado_mt"] == "Perdiendo", "pts_final"] > 0).mean(), 3),
            "empatando_y_gano": r((s.loc[s["estado_mt"] == "Empatando", "pts_final"] == 3).mean(), 3),
            "ganando_y_no_gano": r((s.loc[s["estado_mt"] == "Ganando", "pts_final"] < 3).mean(), 3),
        }
        res_mt[g] = out
    # Diferencia contra la Liga cuando no va ganando al descanso
    ref = mt[(mt["caso"] == "Liga") & (mt["estado_mt"] != "Ganando")]["d_xg_dif"].dropna().values
    for g in casos:
        s = mt[(mt["caso"] == g) & (mt["estado_mt"] != "Ganando")]["d_xg_dif"].dropna().values
        if len(s) >= 5 and len(ref) >= 5:
            obs = s.mean() - ref.mean()
            todo = np.concatenate([s, ref]); c = 0
            for _ in range(5000):
                RNG.shuffle(todo)
                c += abs(todo[:len(s)].mean() - todo[len(s):].mean()) >= abs(obs)
            res_mt[g]["vs_liga_no_ganando"] = {"diferencia": r(obs), "p": r((c + 1) / 5001, 4), "n": int(len(s))}
    salida["medio_tiempo"] = res_mt

    t = []
    for g, o in res_mt.items():
        fila = {"caso": g, "n": o["partidos"]}
        for kpi in ["posesion_pct", "ppda", "field_tilt", "npxg_p90", "npxg_contra_p90"]:
            fila[NOMBRE_KPI[kpi]] = o["kpis"][kpi]["cambio"][0]
        pe = o["por_estado"]
        fila["dXGdif perdiendo"] = (pe.get("Perdiendo") or {}).get("d_xg_dif", [None])[0]
        fila["dXGdif empatando"] = (pe.get("Empatando") or {}).get("d_xg_dif", [None])[0]
        fila["pts 2T perd"] = (pe.get("Perdiendo") or {}).get("pts_ganados_2t")
        fila["pts 2T emp"] = (pe.get("Empatando") or {}).get("pts_ganados_2t")
        fila["vs liga (p)"] = (o.get("vs_liga_no_ganando") or {}).get("p")
        t.append(fila)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.DataFrame(t).set_index("caso").round(3).to_string())

    # ================= 3. JUGADORES =================
    print("\n3. Desarrollo de jugadores (mismos jugadores, por 90 minutos)")
    J = pd.DataFrame([{"caso": c, "jugador": j, **v} for (c, j), v in jug.items()])
    salida["jugadores"] = {}
    if not J.empty:
        m90 = ["npxg", "xa", "pases_prog", "cond_prog", "presiones", "acciones_def", "obv"]
        for c_ in m90:
            J[f"{c_}90"] = J[c_] / J["minutos"] * 90
        J["precision"] = J["pases_ok"] / J["pases"].replace(0, np.nan)
        J["posicion"] = J["jugador"].map(lambda j: max(jug_pos[j], key=jug_pos[j].get) if jug_pos[j] else None)
        hay_obv = J["obv"].abs().sum() > 0
        metr = [f"{c_}90" for c_ in m90 if c_ != "obv" or hay_obv] + ["precision"]
        J = J[J["minutos"] >= J["caso"].map(MIN_MINUTOS)]
        for a, b in [("Ortiz-America", "Jardine-America"), ("Jardine-America", "Almada-America")]:
            A_, B_ = J[J["caso"] == a].set_index("jugador"), J[J["caso"] == b].set_index("jugador")
            comunes = sorted(set(A_.index) & set(B_.index))
            if not comunes:
                continue
            res = {"jugadores": len(comunes), "metricas": {}, "detalle": []}
            for mtr in metr:
                d = (B_.loc[comunes, mtr] - A_.loc[comunes, mtr]).values
                mejora = float((d > 0).mean())
                res["metricas"][mtr] = {"cambio_mediano": r(np.median(d)), "media": boot_media(d)[:3],
                                        "pct_mejora": r(mejora, 3)}
            clave = "obv90" if hay_obv else "pases_prog90"
            for j in comunes:
                res["detalle"].append({"jugador": j, "posicion": A_.loc[j, "posicion"],
                                       "min_a": r(A_.loc[j, "minutos"], 0), "min_b": r(B_.loc[j, "minutos"], 0),
                                       **{f"{mtr}_a": r(A_.loc[j, mtr]) for mtr in metr},
                                       **{f"{mtr}_b": r(B_.loc[j, mtr]) for mtr in metr}})
            res["detalle"].sort(key=lambda x: -((x.get(f"{clave}_b") or 0) - (x.get(f"{clave}_a") or 0)))
            res["clave"] = clave
            salida["jugadores"][f"{a}->{b}"] = res
            print(f"\n  {a} -> {b}: {len(comunes)} jugadores con minutos en ambos")
            for mtr, v in res["metricas"].items():
                print(f"    {mtr:<16} cambio mediano {v['cambio_mediano']:>8}   media {v['media']}   mejoran {v['pct_mejora']}")
            print(f"    Mas suben en {clave}: " + ", ".join(
                f"{x['jugador']} ({(x.get(clave + '_b') or 0) - (x.get(clave + '_a') or 0):+.2f})" for x in res["detalle"][:5]))
            print(f"    Mas bajan en {clave}: " + ", ".join(
                f"{x['jugador']} ({(x.get(clave + '_b') or 0) - (x.get(clave + '_a') or 0):+.2f})" for x in res["detalle"][-5:]))

    ruta = data / "preguntas_extra.json"
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(salida, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"\nGuardado: {ruta} en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
