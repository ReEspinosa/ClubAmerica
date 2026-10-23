# Jardine en Datos

**La historia de un entrenador a través de los datos.** Proyecto para el reto del Club América del Hackathon ISAC 2026 (ITAM Sports Analytics Conference).

Reporte interactivo: **https://reespinosa.github.io/ClubAmerica/**

Medimos el estilo de juego de André Jardine con eventos y fotos 360 de Hudl StatsBomb de 388 partidos de la Liga MX (2021–2026). Comparamos su América con el de Fernando Ortiz, con su propio Atlético de San Luis y con los equipos de Guillermo Almada. El objetivo es separar qué parte del estilo es del entrenador, qué parte es del club y qué parte es azar.

---

## Preguntas y respuestas

| Pregunta | Respuesta corta |
|---|---|
| 1. ¿El estilo es de Jardine o del América? | Mayormente del club: 13 de 20 rasgos los explica el América. Lo que viaja con Jardine es la paciencia sin balón: el rival le da 19.7 pases antes de que presione (rival típico: 16.2). |
| 2. ¿Qué cambió después del tricampeonato? | Hay un punto de cambio en febrero de 2025. Presionó más y tuvo más el balón, pero tiró peor: el xG por tiro bajó de 0.113 a 0.088. El contrafactual 360 indica que no fue por egoísmo: solo en 1.7% de sus tiros había un pase claramente mejor. |
| 3. ¿Ya se nota la mano de Almada? | Todavía poco. El índice de transición es 0.30 (0 = Jardine, 1 = Almada), y la distancia de Wasserstein al América de Jardine (1.5 yardas) es menor que el ruido entre muestras (1.6). Lo que sí cambió es la forma del bloque: presiona más cerca del balón, pero el bloque es casi 3 yardas más largo. |

También evaluamos al entrenador en sí: liguilla contra fase regular, ajustes en el medio tiempo y el rendimiento de los mismos jugadores con distintos entrenadores. Su firma más clara es un plan de dos tiempos: espera en el primero y aprieta en el segundo. Pasa en el América (PPDA de 26.0 a 19.8) y en San Luis (de 23.4 a 15.7), y no pasa con Ortiz ni con Almada.

---

## Estructura del repositorio

```
.
├── index.html          Reporte interactivo (GitHub Pages)
├── src/                Código del análisis (Python)
├── figuras/            Figuras estáticas de las preguntas 1 a 3
├── resultados/         Tablas agregadas (CSV)
├── requirements.txt
└── README.md
```

Los datos de StatsBomb **no** están en este repositorio. Los proporcionó ISAC para uso exclusivo de la competencia y solo pueden publicarse resultados. La carpeta `data/` está en `.gitignore`.

---

## Cómo reproducirlo

Se necesita acceso a la API de StatsBomb. Las credenciales se leen **solo** de variables de entorno; nunca se escriben en el código.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export SB_USERNAME="..."
export SB_PASSWORD="..."
```

Los scripts se corren desde la raíz del repositorio, en este orden:

| Paso | Script | Qué hace |
|---|---|---|
| 1 | `src/verificar_datos.py` | Comprueba el acceso a la API |
| 2 | `src/listar_entrenadores.py` | Lista entrenadores y partidos por equipo |
| 3 | `src/descargar_casos.py` | Descarga eventos, fotos 360 y alineaciones de los seis casos a `data/` |
| 4 | `src/construir_huella.py` | Calcula los 20 rasgos por partido y la rejilla de xT |
| 5 | `src/analisis_preguntas.py` | Preguntas 1 a 3: R² parcial, puntos de cambio, índice de Fisher |
| 6 | `src/minuto_a_minuto.py` | xT minuto a minuto y partido típico por bloques de 15 minutos |
| 7 | `src/analisis_completo.py` | Bloque defensivo 360, transiciones, contexto, redes de pases, sustituciones, balón parado, OBV y Monte Carlo |
| 8 | `src/partido_tipico.py` | Simulación del partido típico y puntos débiles |
| 9 | `src/ideas_extra.py` | Transporte óptimo, control del espacio y fantasma de Jardine |
| 10 | `src/preguntas_extra.py` | Medio tiempo y desarrollo de jugadores |
| 11 | `src/contrafactual_tiros.py` | Contrafactual 360 de cada tiro |

`src/descargar_datos.py` es una versión anterior de la descarga, para un solo entrenador.

---

## Métodos

Coordenadas StatsBomb: cancha de $120 \times 80$ yardas, con el equipo que actúa atacando hacia $x = 120$. Todos los intervalos son de 90% por bootstrap sobre partidos.

### Huella de estilo

Cada partido $m$ se describe con un vector de 20 rasgos $\mathbf{z}_m \in \mathbb{R}^{20}$ (salida, progresión, creación, presión, transiciones), estandarizado contra los rivales de los mismos partidos.

$$\text{PPDA} = \frac{\text{pases del rival en su 60\% defensivo}}{\text{acciones defensivas propias en esa zona}}$$

### Expected Threat como cadena de Markov

La cancha se divide en $16 \times 12$ zonas. El valor de cada zona resuelve el sistema

$$xT(i) = s_i\, g_i + m_i \sum_j T_{ij}\, xT(j),$$

donde $s_i$ y $m_i$ son las probabilidades de tirar y de mover el balón, $T_{ij}$ es la matriz de transición y $g_i$ la probabilidad de gol, suavizada con un prior bayesiano en zonas con pocos tiros.

### Predictibilidad: tasa de entropía

$$h = -\sum_i \pi_i \sum_j P_{ij} \log_2 P_{ij},$$

con $\pi$ la distribución estacionaria de la cadena de zonas. Más bits significa más rutas distintas para avanzar.

### Entrenador contra club: R² parcial

Para cada rasgo $y$:

$$y = \mu + \alpha_{\text{entrenador}} + \beta_{\text{club}} + \gamma\,\text{local} + \boldsymbol{\delta}^\top \text{marcador} + \varepsilon.$$

El aporte de cada factor es la caída de $R^2$ al quitarlo del modelo.

### Puntos de cambio

Segmentación binaria sobre la serie $\mathbf{z}_1, \dots, \mathbf{z}_n$. Se acepta un corte en $t$ solo si

$$C(a,b) - C(a,t) - C(t,b) > d \ln n,$$

con $C$ el costo cuadrático del segmento y una penalización tipo BIC. Los cambios se confirman con pruebas de permutación.

### Índice de transición (discriminante de Fisher)

$$\mathbf{w} = \Sigma^{-1}(\boldsymbol{\mu}_A - \boldsymbol{\mu}_J), \qquad T(\mathbf{z}) = \frac{(\mathbf{z} - \boldsymbol{\mu}_J)^\top \mathbf{w}}{(\boldsymbol{\mu}_A - \boldsymbol{\mu}_J)^\top \mathbf{w}},$$

de modo que $T = 0$ es el América promedio de Jardine y $T = 1$ el equipo promedio de Almada.

### Transporte óptimo

Los repartos de acciones $a$ y $b$ sobre una rejilla de $12 \times 8$ se comparan con la distancia de Wasserstein-1:

$$W(a,b) = \min_{\gamma \ge 0} \sum_{i,j} \gamma_{ij}\, d_{ij} \quad \text{s.a.} \quad \sum_j \gamma_{ij} = a_i,\; \sum_i \gamma_{ij} = b_j.$$

El ruido se estima con 200 submuestras de partidos de Jardine del mismo tamaño que la muestra de Almada.

### Control del espacio (Voronoi en 360)

Para cada foto 360 con el rival en posesión y cada punto $z$ de una rejilla de 3 yardas visible para la cámara:

$$P(\text{rival controla } z) = \mathbb{E}\Big[\mathbf{1}\big\{\min_{a \in A} \lVert z - a \rVert < \min_{d \in D} \lVert z - d \rVert\big\}\Big].$$

### El fantasma de Jardine (ghosting por vecino más cercano)

Para cada foto del América de Almada se buscan fotos de Jardine con el balón a menos de 6 yardas y se elige la de atacantes más parecidos según la distancia de Chamfer:

$$d(P,Q) = \tfrac{1}{2}\Big[\tfrac{1}{|P|}\sum_{p}\min_{q}\lVert p-q\rVert + \tfrac{1}{|Q|}\sum_{q}\min_{p}\lVert p-q\rVert\Big].$$

Los defensores se emparejan con el algoritmo húngaro para medir cuánto se mueve cada uno. El ruido se mide comparando a Jardine contra su propio fantasma, tomado de otros partidos suyos.

### Simulación Monte Carlo

Cada partido se rejuega 20,000 veces con sus tiros reales. Para no contar dos veces los rebotes, cada posesión con tiro es gol con probabilidad

$$P(\text{gol}) = 1 - \prod_k (1 - xG_k), \qquad xPts = 3\,P(\text{ganar}) + P(\text{empatar}).$$

### Contrafactual 360

En cada tiro se evalúa a cada compañero visible con un modelo $\operatorname{logit}(xG) = \beta_0 + \beta_1 d + \beta_2 \theta + \beta_3 \ln(1+d)$, ajustado sobre 10,316 tiros, donde $d$ es la distancia y $\theta$ el ángulo a la portería. Ese xG se castiga por los rivales que tapan el tiro. Hay mejor opción si el pase era posible y el xG del compañero supera al del tirador por 0.05 o más.

---

## Limitaciones

- "Club" incluye plantilla y época: el modelo separa asociaciones ajustadas, no causas.
- Almada lleva 11 partidos con el América; la pregunta 3 son señales tempranas.
- Las fotos 360 no traen velocidades y solo muestran lo que capta la cámara.
- En liguilla los rivales son más fuertes; parte del cambio de plan puede deberse a eso.

---

## Aviso

Análisis independiente de un equipo participante del Hackathon ISAC 2026; no es una publicación oficial del Club América. Datos: Hudl StatsBomb, proporcionados por ISAC para uso exclusivo de la competencia. Este repositorio contiene solo código y resultados agregados. Las fotografías y la identidad del club pertenecen a sus titulares.

Autora: Rebeca Espinosa Roque ([@ReEspinosa](https://github.com/ReEspinosa)).
