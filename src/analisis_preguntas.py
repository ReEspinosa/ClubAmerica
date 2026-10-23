"""
Pasos 3, 4 y 5 - Analisis de las tres preguntas (Hackathon ISAC 2026).

Entradas:
    data/huella.csv     (de construir_huella.py)
    data/partidos.csv   (de descargar_casos.py)

Salidas:
    figuras/*.png       graficas listas para el reporte
    resultados/*.csv    tablas con los numeros de cada grafica

Preguntas:
    P1  El estilo es de Jardine o del America?
    P2  Como cambio el estilo de Jardine cuando el equipo dejo de ganar titulos?
    P3  Almada en el America: ya se nota su mano?

Uso:
    pip install matplotlib
    python analisis_preguntas.py
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(2026)
N_BOOT = 2000

# ---------------------------------------------------------------------------
# Indicadores que entran al analisis (deben existir en huella.csv)
# signo = +1 si "mas alto" significa mas de lo que dice la etiqueta
# ---------------------------------------------------------------------------
KPIS = {
    "posesion_pct":                 "Posesion",
    "salida_corta_portero":         "Salida corta del portero",
    "pases_largos_pct":             "Pases largos",
    "pases_progresivos_p90":        "Pases progresivos",
    "conducciones_progresivas_p90": "Conducciones progresivas",
    "xt_por_posesion":              "xT por posesion",
    "carril_izq_pct":               "Ataque por izquierda",
    "carril_centro_pct":            "Ataque por el centro",
    "carril_der_pct":               "Ataque por derecha",
    "npxg_p90":                     "xG a favor",
    "xg_por_tiro":                  "Calidad de tiro",
    "tiros_transicion_pct":         "Tiros en transicion",
    "intensidad_presion":           "Intensidad de presion (-PPDA)",
    "presion_alta_pct":             "Presion alta",
    "contrapresion_por_posesion":   "Contrapresion",
    "altura_recuperacion":          "Altura de recuperacion",
    "npxg_contra_p90":              "xG en contra",
    "field_tilt":                   "Field tilt",
    "entropia_salida":              "Variedad en la salida",
    "entropia_progresion":          "Variedad en la progresion",
}

FINALES = {  # finales de Liga MX del America con Jardine (segun el calendario)
    "2023-12-18": "Final vs Tigres",
    "2024-05-27": "Final vs Cruz Azul",
    "2024-12-16": "Final vs Monterrey",
    "2025-05-26": "Final vs Toluca",
}
FIN_TRICAMPEONATO = "2024-12-16"

COLORES = {
    "Jardine-America": "#c8a200", "Jardine-SanLuis": "#7a6a2e",
    "Ortiz-America": "#2f6f9f", "Almada-Pachuca": "#3a8f5c",
    "Almada-Santos": "#88b39a", "Almada-America": "#b5422f", "Liga": "#9a9a9a",
}
ORDEN = ["Ortiz-America", "Jardine-SanLuis", "Jardine-America",
         "Almada-Santos", "Almada-Pachuca", "Almada-America"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11,
})


# ---------------------------------------------------------------------------
# Preparacion
# ---------------------------------------------------------------------------
def cargar(data):
    h = pd.read_csv(data / "huella.csv")
    p = pd.read_csv(data / "partidos.csv")
    h["match_date"] = pd.to_datetime(h["match_date"])

    # La intensidad de presion es -PPDA (asi "mas alto" = presiona mas)
    h["intensidad_presion"] = -h["ppda"].clip(upper=60)

    # Localia y club para las filas de los casos
    loc = p.set_index(["match_id", "equipo"])["es_local"]
    h["es_local"] = [loc.get((m, e), np.nan) for m, e in zip(h["match_id"], h["equipo"])]
    h["club"] = h["equipo"]
    h["entrenador"] = h["caso"].str.split("-").str[0].where(h["caso"] != "Liga")

    # Proporcion del partido en cada estado de marcador
    tot = h[["min_ganando", "min_empatando", "min_perdiendo"]].sum(axis=1).replace(0, np.nan)
    h["frac_ganando"] = h["min_ganando"] / tot
    h["frac_perdiendo"] = h["min_perdiendo"] / tot

    # z-scores contra la referencia de liga
    liga = h[h["caso"] == "Liga"]
    for k in KPIS:
        mu, sd = liga[k].mean(), liga[k].std()
        h[f"z_{k}"] = (h[k] - mu) / (sd if sd > 0 else 1)
    return h


def zcols():
    return [f"z_{k}" for k in KPIS]


def bootstrap_media(x, n=N_BOOT):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return x.mean(), np.percentile(m, 5), np.percentile(m, 95)


def perfil(df):
    return df[zcols()].mean()


def distancia(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(((a[ok] - b[ok]) ** 2).mean())) if ok.any() else np.nan


# ---------------------------------------------------------------------------
# ADN por caso
# ---------------------------------------------------------------------------
def adn_por_caso(h, figs, res):
    filas = []
    for caso in ORDEN + ["Liga"]:
        d = h[h["caso"] == caso]
        if d.empty:
            continue
        for k, nombre in KPIS.items():
            m, lo, hi = bootstrap_media(d[f"z_{k}"])
            filas.append({"caso": caso, "indicador": nombre, "kpi": k,
                          "z": m, "ic90_bajo": lo, "ic90_alto": hi,
                          "valor_crudo": d[k].mean(), "partidos": d["match_id"].nunique()})
    t = pd.DataFrame(filas)
    t.to_csv(res / "adn_por_caso.csv", index=False)

    piv = t.pivot(index="indicador", columns="caso", values="z")
    piv = piv[[c for c in ORDEN if c in piv.columns]].loc[list(KPIS.values())]
    fig, ax = plt.subplots(figsize=(7.5, 8))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [c.replace("-", "\n") for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.9 else "black")
    ax.set_title("ADN de cada caso (desviaciones vs. la referencia de liga)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="z (0 = liga)")
    fig.savefig(figs / "01_adn_por_caso.png")
    plt.close(fig)
    return t


# ---------------------------------------------------------------------------
# P1: entrenador o club
# ---------------------------------------------------------------------------
def ols_r2(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1 - resid.var() / y.var(), beta


def dummies(serie, base):
    d = pd.get_dummies(serie).astype(float)
    return d.drop(columns=[base], errors="ignore")


def pregunta1(h, figs, res):
    casos = h[h["caso"] != "Liga"].copy()
    perfiles = {c: perfil(casos[casos["caso"] == c]) for c in ORDEN if (casos["caso"] == c).any()}

    # Matriz de distancias entre perfiles
    nombres = list(perfiles)
    D = pd.DataFrame([[distancia(perfiles[a], perfiles[b]) for b in nombres] for a in nombres],
                     index=nombres, columns=nombres)
    D.to_csv(res / "p1_distancias_entre_casos.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(D.values, cmap="viridis_r")
    ax.set_xticks(range(len(nombres)), [n.replace("-", "\n") for n in nombres], fontsize=8)
    ax.set_yticks(range(len(nombres)), nombres, fontsize=8)
    for i in range(len(nombres)):
        for j in range(len(nombres)):
            ax.text(j, i, f"{D.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if D.values[i, j] > D.values.max() * 0.6 else "black")
    ax.set_title("Que tan distintos son los estilos\n(distancia entre perfiles; 0 = identicos)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(figs / "02_p1_distancias.png")
    plt.close(fig)

    # Regresion por indicador: entrenador + club + contexto
    X_ctx = casos[["es_local", "frac_ganando", "frac_perdiendo"]].astype(float).fillna(0.5)
    E = dummies(casos["entrenador"], "Jardine")
    C = dummies(casos["club"], "América")
    filas = []
    for k, nombre in KPIS.items():
        y = casos[f"z_{k}"].values
        ok = ~np.isnan(y)
        if ok.sum() < 30:
            continue
        uno = np.ones((ok.sum(), 1))
        full = np.hstack([uno, X_ctx.values[ok], E.values[ok], C.values[ok]])
        sin_e = np.hstack([uno, X_ctx.values[ok], C.values[ok]])
        sin_c = np.hstack([uno, X_ctx.values[ok], E.values[ok]])
        r2, _ = ols_r2(y[ok], full)
        r2_e, _ = ols_r2(y[ok], sin_e)
        r2_c, _ = ols_r2(y[ok], sin_c)
        aporte_e, aporte_c = max(r2 - r2_e, 0), max(r2 - r2_c, 0)

        # Bootstrap por partidos para la parte de entrenador vs club
        idx_ok = np.where(ok)[0]
        dif = []
        for _ in range(400):
            s = RNG.choice(idx_ok, size=len(idx_ok), replace=True)
            u = np.ones((len(s), 1))
            f_ = np.hstack([u, X_ctx.values[s], E.values[s], C.values[s]])
            a, _ = ols_r2(y[s], f_)
            b, _ = ols_r2(y[s], np.hstack([u, X_ctx.values[s], C.values[s]]))
            c, _ = ols_r2(y[s], np.hstack([u, X_ctx.values[s], E.values[s]]))
            dif.append(max(a - b, 0) - max(a - c, 0))
        lo, hi = np.percentile(dif, [5, 95])
        if lo > 0:
            veredicto = "Entrenador"
        elif hi < 0:
            veredicto = "Club"
        else:
            veredicto = "Compartido / no concluyente"
        filas.append({"indicador": nombre, "kpi": k, "r2_total": r2,
                      "aporte_entrenador": aporte_e, "aporte_club": aporte_c,
                      "dif_ic90_bajo": lo, "dif_ic90_alto": hi, "veredicto": veredicto})
    t = pd.DataFrame(filas)
    t["balance"] = t["aporte_entrenador"] - t["aporte_club"]
    t = t.sort_values("balance", ascending=True)  # arriba: rasgos del entrenador
    t.to_csv(res / "p1_entrenador_vs_club.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    yy = np.arange(len(t))
    ax.barh(yy - 0.2, t["aporte_entrenador"], 0.4, label="Entrenador", color="#c8a200")
    ax.barh(yy + 0.2, t["aporte_club"], 0.4, label="Club / plantilla", color="#2f6f9f")
    etiquetas = [f"{n}  [{v}]" if v != "Compartido / no concluyente" else n
                 for n, v in zip(t["indicador"], t["veredicto"])]
    ax.set_yticks(yy, etiquetas, fontsize=8)
    ax.set_xlabel("Varianza que explica cada factor (R2 parcial), ya ajustado por localia y marcador")
    ax.set_title("De quien es cada rasgo: del entrenador o del club?")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(figs / "03_p1_entrenador_vs_club.png")
    plt.close(fig)
    return D, t


# ---------------------------------------------------------------------------
# P2: evolucion de Jardine
# ---------------------------------------------------------------------------
def segmentacion_binaria(Z, min_seg=8, max_cambios=3):
    """Puntos de cambio en la media de un vector (z), penalizacion tipo BIC."""
    Z = np.nan_to_num(Z)
    n, d = Z.shape
    pen = d * np.log(n)

    def costo(a, b):
        s = Z[a:b]
        return ((s - s.mean(0)) ** 2).sum()

    cortes, segs = [], [(0, n)]
    for _ in range(max_cambios):
        mejor = None
        for a, b in segs:
            base = costo(a, b)
            for t in range(a + min_seg, b - min_seg + 1):
                g = base - costo(a, t) - costo(t, b)
                if g > pen and (mejor is None or g > mejor[0]):
                    mejor = (g, t, (a, b))
        if mejor is None:
            break
        _, t, (a, b) = mejor
        cortes.append(t)
        segs.remove((a, b))
        segs += [(a, t), (t, b)]
    return sorted(cortes)


def pregunta2(h, figs, res):
    j = h[h["caso"] == "Jardine-America"].sort_values("match_date").reset_index(drop=True)
    cortes = segmentacion_binaria(j[zcols()].values)
    fechas_corte = [j.loc[c, "match_date"] for c in cortes]
    pd.DataFrame({"indice_partido": cortes,
                  "fecha": [f.date() for f in fechas_corte],
                  "rival": [j.loc[c, "rival"] for c in cortes]}).to_csv(
        res / "p2_puntos_de_cambio.csv", index=False)

    # Serie de tiempo de los indicadores clave
    clave = ["ppda", "presion_alta_pct", "contrapresion_por_posesion",
             "altura_recuperacion", "entropia_salida", "entropia_progresion",
             "npxg_p90", "npxg_contra_p90"]
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
    for ax, k in zip(axes.flat, clave):
        ax.plot(j["match_date"], j[k], ".", color="#c8a200", alpha=0.35, ms=4)
        ax.plot(j["match_date"], j[k].rolling(5, min_periods=3, center=True).mean(),
                color="#c8a200", lw=2)
        for f in FINALES:
            ax.axvline(pd.Timestamp(f), color="#999", lw=0.8, ls=":")
        for f in fechas_corte:
            ax.axvline(f, color="#b5422f", lw=1.4)
        if k == "ppda":
            ax.invert_yaxis()
            ax.set_title("PPDA (eje invertido: arriba = presiona mas)", fontsize=9)
        else:
            ax.set_title(KPIS[k], fontsize=9)
    axes[0, 0].plot([], [], ":", color="#999", label="Finales")
    axes[0, 0].plot([], [], color="#b5422f", label="Cambio detectado")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("Jardine en el America: evolucion partido a partido (promedio movil de 5)",
                 fontweight="bold")
    fig.autofmt_xdate()
    fig.savefig(figs / "04_p2_evolucion_jardine.png")
    plt.close(fig)

    # Tricampeonato vs despues
    j["etapa"] = np.where(j["match_date"] <= pd.Timestamp(FIN_TRICAMPEONATO),
                          "Tricampeonato", "Despues")
    a = j[j["etapa"] == "Tricampeonato"]
    b = j[j["etapa"] == "Despues"]
    filas = []
    for k, nombre in KPIS.items():
        xa, xb = a[f"z_{k}"].dropna().values, b[f"z_{k}"].dropna().values
        if len(xa) < 5 or len(xb) < 5:
            continue
        dif = xb.mean() - xa.mean()
        boots = [RNG.choice(xb, len(xb)).mean() - RNG.choice(xa, len(xa)).mean()
                 for _ in range(N_BOOT)]
        lo, hi = np.percentile(boots, [5, 95])
        pool = np.concatenate([xa, xb])
        perm = []
        for _ in range(N_BOOT):
            RNG.shuffle(pool)
            perm.append(pool[len(xa):].mean() - pool[:len(xa)].mean())
        p = (np.abs(perm) >= abs(dif)).mean()
        filas.append({"indicador": nombre, "kpi": k,
                      "tricampeonato_crudo": a[k].mean(), "despues_crudo": b[k].mean(),
                      "cambio_z": dif, "ic90_bajo": lo, "ic90_alto": hi, "p_permutacion": p})
    t = pd.DataFrame(filas).sort_values("cambio_z")
    t.to_csv(res / "p2_tricampeonato_vs_despues.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    yy = np.arange(len(t))
    sig = (t["ic90_bajo"] > 0) | (t["ic90_alto"] < 0)
    ax.errorbar(t["cambio_z"], yy, xerr=[t["cambio_z"] - t["ic90_bajo"], t["ic90_alto"] - t["cambio_z"]],
                fmt="none", ecolor="#bbb", lw=1.5)
    ax.scatter(t["cambio_z"], yy, c=np.where(sig, "#b5422f", "#777"), zorder=3, s=28)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(yy, t["indicador"], fontsize=8)
    ax.set_xlabel("Cambio despues del tricampeonato (en desviaciones de liga), IC 90%")
    ax.set_title("Que cambio cuando el America dejo de ganar titulos?\n(rojo = cambio claro)")
    fig.savefig(figs / "05_p2_tricampeonato_vs_despues.png")
    plt.close(fig)
    return cortes, j, t


# ---------------------------------------------------------------------------
# P3: indice de transicion de Almada
# ---------------------------------------------------------------------------
def pregunta3(h, figs, res):
    grupo_j = h[h["caso"] == "Jardine-America"]
    grupo_a = h[h["caso"].isin(["Almada-Pachuca", "Almada-Santos"])]
    adn_j, adn_a = perfil(grupo_j), perfil(grupo_a)

    # Eje discriminante de Fisher: w = (mu_A - mu_J) / varianza dentro de grupos.
    # Pesa mas los rasgos donde Jardine y Almada difieren mucho y varian poco.
    diff = (adn_a - adn_j).values.astype(float)
    var = ((grupo_j[zcols()].var() * (len(grupo_j) - 1) + grupo_a[zcols()].var() * (len(grupo_a) - 1))
           / (len(grupo_j) + len(grupo_a) - 2)).values.astype(float)
    w = diff / np.where(var > 0, var, np.nan)
    mu_j = adn_j.values.astype(float)

    def indice(fila):
        """0 = perfil medio de Jardine, 1 = perfil medio de Almada (proyeccion)."""
        z = fila[zcols()].values.astype(float)
        ok = ~(np.isnan(z) | np.isnan(w))
        den = (diff[ok] * w[ok]).sum()
        return ((z[ok] - mu_j[ok]) * w[ok]).sum() / den if den > 0 else np.nan

    h = h.copy()
    sel = h[h["caso"].isin(["Jardine-America", "Almada-Pachuca", "Almada-Santos", "Almada-America"])].copy()
    sel["indice_transicion"] = sel.apply(indice, axis=1)
    sel[["caso", "match_id", "match_date", "rival", "indice_transicion"]].to_csv(
        res / "p3_indice_transicion.csv", index=False)

    am = sel[sel["caso"] == "Almada-America"].sort_values("match_date")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for caso, color, y0 in [("Jardine-America", COLORES["Jardine-America"], "ADN Jardine"),
                            ("Almada-Pachuca", COLORES["Almada-Pachuca"], "ADN Almada")]:
        v = sel.loc[sel["caso"] == caso, "indice_transicion"]
        lo, hi = v.quantile([0.25, 0.75])
        ax.axhspan(lo, hi, color=color, alpha=0.18, lw=0)
        ax.text(1.01, v.median(), f"{y0}\n(50% central)", transform=ax.get_yaxis_transform(),
                color=color, fontsize=8, va="center")
    ax.plot(range(len(am)), am["indice_transicion"], "o-", color=COLORES["Almada-America"], lw=1.5)
    ax.set_xticks(range(len(am)), [f"{d:%d %b}\n{r[:10]}" for d, r in zip(am["match_date"], am["rival"])],
                  fontsize=7)
    ax.axhline(0, color=COLORES["Jardine-America"], lw=0.8)
    ax.axhline(1, color=COLORES["Almada-Pachuca"], lw=0.8)
    ax.set_ylim(-1, 2)
    ax.set_ylabel("Indice de transicion\n0 = Jardine promedio, 1 = Almada promedio")
    ax.set_title("Almada en el America: partido a partido, a quien se parece?")
    fig.savefig(figs / "06_p3_indice_transicion.png")
    plt.close(fig)

    # Por indicador: donde esta el America de Almada entre los dos ADN
    p_am = perfil(h[h["caso"] == "Almada-America"])
    filas = []
    for k, nombre in KPIS.items():
        zj, za, zm = adn_j[f"z_{k}"], adn_a[f"z_{k}"], p_am[f"z_{k}"]
        separacion = za - zj
        pos = (zm - zj) / separacion if abs(separacion) >= 0.3 else np.nan
        filas.append({"indicador": nombre, "kpi": k, "z_jardine": zj, "z_almada_adn": za,
                      "z_almada_america": zm, "separacion": separacion, "posicion_0J_1A": pos})
    t = pd.DataFrame(filas)
    t.to_csv(res / "p3_rasgos_que_cambiaron.csv", index=False)

    tt = t.dropna(subset=["posicion_0J_1A"]).sort_values("posicion_0J_1A")
    if len(tt):
        fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(tt) + 1.5))
        yy = np.arange(len(tt))
        ax.axvspan(-0.5, 0.33, color=COLORES["Jardine-America"], alpha=0.12, lw=0)
        ax.axvspan(0.67, 1.5, color=COLORES["Almada-Pachuca"], alpha=0.12, lw=0)
        ax.scatter(tt["posicion_0J_1A"].clip(-0.5, 1.5), yy, color=COLORES["Almada-America"], zorder=3)
        ax.set_yticks(yy, tt["indicador"], fontsize=8)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 0.5, 1], ["Como Jardine", "A la mitad", "Como Almada"])
        ax.set_xlabel("Un punto pegado al borde va todavia mas alla que ese entrenador", fontsize=8)
        ax.set_title("Que rasgos de Almada ya se ven en el America?\n"
                     "(solo indicadores donde Jardine y Almada difieren al menos 0.3 sd)")
        fig.savefig(figs / "07_p3_rasgos.png")
        plt.close(fig)
    return sel, t


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--salida", default=".")
    args = ap.parse_args()
    data, base = Path(args.data), Path(args.salida)
    figs, res = base / "figuras", base / "resultados"
    figs.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)

    h = cargar(data)
    print(f"Huella cargada: {len(h)} filas, casos: {sorted(h['caso'].unique())}")

    adn = adn_por_caso(h, figs, res)

    D, p1 = pregunta1(h, figs, res)
    print("\n=== P1. Estilo de Jardine o del America? ===")
    par = lambda a, b: D.loc[a, b] if a in D.index and b in D.columns else np.nan
    print(f"  Distancia Jardine-America vs Jardine-SanLuis : {par('Jardine-America', 'Jardine-SanLuis'):.2f}")
    print(f"  Distancia Jardine-America vs Ortiz-America   : {par('Jardine-America', 'Ortiz-America'):.2f}")
    print(f"  Distancia Almada-America  vs Almada-Pachuca  : {par('Almada-America', 'Almada-Pachuca'):.2f}")
    print(f"  Distancia Almada-America  vs Jardine-America : {par('Almada-America', 'Jardine-America'):.2f}")
    print("  Rasgos por veredicto:")
    for v, g in p1.groupby("veredicto"):
        print(f"    {v}: {', '.join(g['indicador'])}")

    cortes, j, p2 = pregunta2(h, figs, res)
    print("\n=== P2. Evolucion de Jardine ===")
    print(f"  Puntos de cambio detectados: "
          f"{[str(j.loc[c, 'match_date'].date()) for c in cortes] or 'ninguno'}")
    claros = p2[(p2["ic90_bajo"] > 0) | (p2["ic90_alto"] < 0)]
    for r in claros.itertuples():
        sentido = "subio" if r.cambio_z > 0 else "bajo"
        print(f"  {r.indicador}: {sentido} {abs(r.cambio_z):.2f} sd "
              f"({r.tricampeonato_crudo:.3f} -> {r.despues_crudo:.3f}), p={r.p_permutacion:.3f}")
    if claros.empty:
        print("  Ningun cambio claro con IC 90%.")

    sel, p3 = pregunta3(h, figs, res)
    print("\n=== P3. Almada en el America ===")
    for caso in ["Jardine-America", "Almada-Pachuca", "Almada-America"]:
        v = sel.loc[sel["caso"] == caso, "indice_transicion"]
        print(f"  Indice medio {caso:16s}: {v.mean():.2f}  (n={v.notna().sum()})")
    ya = p3[p3["posicion_0J_1A"] >= 0.67]["indicador"].tolist()
    aun = p3[p3["posicion_0J_1A"] <= 0.33]["indicador"].tolist()
    print(f"  Ya se parece a Almada en: {', '.join(ya) or '-'}")
    print(f"  Sigue como Jardine en:    {', '.join(aun) or '-'}")

    print(f"\nFiguras en {figs}/ y tablas en {res}/")


if __name__ == "__main__":
    main()
