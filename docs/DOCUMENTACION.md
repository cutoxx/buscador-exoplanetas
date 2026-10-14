# Documentación técnica

> Volver al [README principal](../README.md).

Busca planetas en datos reales del satélite **TESS** de la NASA con el **método de tránsito**: cuando un planeta pasa frente a su estrella, el brillo baja un poco, siempre con el mismo período.

Para cada señal, el script **no dice "es un planeta"**. La clasifica y explica por qué:

| Veredicto | Significado |
|---|---|
| `CANDIDATO PLANETARIO` | Pasa todas las pruebas. Igual falta confirmarlo con otro método. |
| `PROBABLE FALSO POSITIVO` | Binaria eclipsante, variabilidad de la estrella, eco de otra señal, etc. |
| `DATOS INSUFICIENTES` | Menos de 3 tránsitos con datos, o no hay zonas libres para medir el ruido. |
| `SIN DETECCIÓN` | La señal no se distingue del ruido. |

## Instalación

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # scikit-learn fijo en 1.9.1: la versión con que se entrenó el modelo
```

> Con Python 3.13, `lightkurve` puede fallar al instalarse fuera de un entorno virtual. Dentro de uno funciona.

## Uso

```bash
python buscador_transitos.py --demo planeta          # estrella simulada (también: multi, binaria, ruido)
python buscador_transitos.py --estrella "Pi Mensae"
python buscador_transitos.py --estrella "TIC 259377017" --sectores 3   # TOI-270, 3 sectores
python buscador_transitos.py --estrella "TIC 100100827" --json resultado.json   # WASP-18
python buscador_transitos.py --estrella "Pi Mensae" --clasificador ml/clasificador_toi.joblib   # + IA
```

| Opción | Qué hace |
|---|---|
| `--sectores N` | Cuántos sectores de TESS usar (0 = todos). Más sectores permiten encontrar períodos más largos. |
| `--max-senales N` | Cuántas señales buscar una tras otra (por defecto 3). |
| `--ventana D` | Ventana de limpieza en días (por defecto se elige sola según la estrella). |
| `--salida archivo.png` | Dónde guardar el gráfico. |
| `--clasificador ruta` | Agrega la probabilidad de ser planeta según el modelo de IA (ver abajo). |

## Cómo funciona

1. **Descarga:** primero intenta los datos de 2 min (SPOC) y, si no hay, las imágenes completas (TESS-SPOC, QLP). Lee el radio y la masa de la estrella del catálogo TIC.
2. **Pre-blanqueo** (prewhitening): si la estrella rota y tiene manchas, detecta ese período con Lomb-Scargle y resta un modelo de ondas. Ignora los puntos bajos para no borrar tránsitos.
3. **Limpieza con biweight:** un promedio robusto en ventana móvil (Hippke et al. 2019) que ignora lo que se sale de lo normal. La ventana se elige según la estrella.
4. **Búsqueda BLS** (Box Least Squares): prueba miles de períodos con una grilla uniforme en frecuencia y, para cada período, **solo las duraciones físicamente posibles** según la 3ª ley de Kepler. Si el mejor pico es un armónico (2P o 3P), lo corrige.
5. **Medición honesta del error:** mide cada tránsito contra una recta local, y aplica exactamente el mismo cálculo en ~150 momentos al azar donde no hay nada. Así el error incluye el ruido real (blanco, rojo y residuos), sin suponer nada.
6. **Pruebas** (vetting):

| Prueba | Qué detecta |
|---|---|
| SDE ≥ 10 **y** SNR ≥ 7,1 | Que la señal no sea ruido (calibrado con estrellas simuladas y reales) |
| Mínimo 3 tránsitos con datos | Con 2 eventos cualquier par de ruidos define un período (criterio de Kepler) |
| Tránsitos al borde de huecos de datos | Rampas instrumentales de TESS al reiniciar cada órbita |
| Duración demasiado corta con forma de caja | Destellos de la estrella y picos instrumentales |
| Pares vs impares (descarta > 4σ, advierte > 3σ) | Binaria con el doble del período |
| Eclipse secundario en fase 0,5 (descarta > 5σ, advierte > 3σ) | Binaria (con excepción física para Júpiter muy calientes, que brillan) |
| Mismo período u otro múltiplo exacto en otra fase | Eclipse secundario de una binaria excéntrica |
| Radio > 2,2 R♃, o que tape > 40 % del diámetro estelar | Demasiado grande para un planeta |
| Duración vs 3ª ley de Kepler | Tránsito imposiblemente largo |
| Consistencia entre tránsitos / tránsito faltante | Alias de período o ruido instrumental |
| Onda vs caja | Variabilidad estelar |
| Ajuste de trapecio (fondo plano) | Forma de V → binaria rasante (advertencia) |
| Paridad de épocas / huecos | Período ambiguo (P/2 o 2P), lo advierte |
| SNR < 10 | Señal débil (advertencia): en datos reales, muchas de este nivel son ruido |

7. **Búsqueda iterativa:** tapa los tránsitos encontrados y busca de nuevo, para dar con sistemas de varios planetas.

## Validación

Se usaron tres tipos de pruebas: simulaciones donde se conoce la respuesta (`pruebas_validacion.py`, 242 casos), estrellas reales con planetas conocidos (`validar_reales.py`) y una **prueba nula con datos reales** (240 estrellas sin objetos conocidos, la mitad con el brillo invertido, con `buscar_masivo.py --invertir`).

### Datos simulados

| Prueba | Resultado |
|---|---|
| Estrellas **sin planeta** (ruido blanco y rojo, rotadores rápidos, cadencias de 2/10/30 min) | **0 de 80** dieron "candidato" (tasa de falsas alarmas < 3,7 % con 95 % de confianza) |
| Planetas con SNR ≥ 10 | **65 de 69** recuperados como candidatos (94 %). Los que faltan son sobre todo de período largo con solo 2 tránsitos en un sector: salen como "datos insuficientes, usar más sectores" |
| Planetas con SNR 7-10 | 2 de 5 (40 %) |
| Planetas con SNR < 7 | 1 de 46 (por debajo del límite de detección, como se espera) |
| Por período (SNR ≥ 15) | P < 4 d: 31/31 · P ≥ 4 d: 17/18 |
| Profundidad medida / real | mediana 0,974 (sesgo −2,6 %) |
| Binarias circulares, excéntricas y gemelas | **24 de 24** descartadas |
| Binarias rasantes (en V) | 6 de 8 descartadas (1 de las 2 restantes con advertencia) |
| Binarias gemelas rasantes | 0 de 8 descartadas, pero 7 de 8 con advertencia de forma en V (ver límites) |

### Prueba nula con datos reales (lo más importante para buscar planetas nuevos)

La simulación no tiene todas las trampas del ruido real de TESS. Con los umbrales originales, 40 estrellas reales al azar dieron 3 "candidatos nuevos", y **con el brillo invertido** (donde no puede haber tránsitos) también salieron 3. Eran ruido. Tras agregar las pruebas de 3 tránsitos, bordes de datos y duración corta:

| Lote (sectores 10 y 20, magnitud 9-12, sin objetos conocidos) | Estrellas | Candidatos |
|---|---|---|
| Brillo invertido (toda señal es falsa alarma) — antes | 120 | 3 (2,5 %) |
| Brillo invertido — **ahora** | 120 | **0** (< 2,5 % con 95 % de confianza) |
| Brillo normal — ahora | 120 | 1 (TIC 38761739, SNR 7,9, con advertencia de señal débil; al verlo parece un efecto instrumental) |

El costo: la sensibilidad simulada bajó de 100 % a 94 % (SNR ≥ 10). Para buscar planetas nuevos es el cambio correcto: un buscador que "encuentra" ruido no sirve.

### Estrellas reales (TESS)

| Estrella | Esperado | Resultado | Radio medido vs publicado |
|---|---|---|---|
| Pi Mensae | planeta c, P = 6,2679 d | ✅ candidato, P = 6,2676 d | 1,98 vs 2,04 R⊕ |
| TOI-270 (3 sectores) | 3 planetas | ✅ los 3 candidatos con período exacto. Con 1 solo sector solo sale b: c y d tienen 2 tránsitos y quedan como "datos insuficientes" | b 1,06 / c 2,37 / d 2,08 vs ~1,25 / 2,42 / 2,13 R⊕ |
| WASP-18 | Júpiter caliente con eclipse secundario real | ✅ candidato; su secundario se identificó como "luz propia del planeta" | 1,26 vs ~1,2 R♃ |
| CM Draconis | binaria eclipsante | ✅ falso positivo ("taparía el 59 % del diámetro de la estrella") | — |
| Pi Mensae con 3 sectores | candidato | ✅ (una versión anterior lo descartaba por un "secundario" de 38 ppm a 3,1σ: era ruido; por eso ahora se descarta recién sobre 5σ) | — |

## Búsqueda masiva (`buscar_masivo.py`)

Analiza muchas estrellas sola, separa lo conocido de lo nuevo y **guarda el avance después de cada estrella**. Si se corta la luz, apagas el PC o lo detienes con Ctrl+C, vuelves a ejecutar **el mismo comando** y retoma donde quedó (probado matando el proceso a mitad de un lote de 40: retomó en la 22, sin repetir ninguna).

```bash
# 100 estrellas al azar del sector 20, magnitud 9 a 12, que no tengan nada conocido
python buscar_masivo.py --sector 20 --max 100 --tmag 9 12 --sin-conocidos --salida busqueda_s20

# Tu propia lista (un TIC o nombre por línea)
python buscar_masivo.py --lista mis_estrellas.txt --salida mi_busqueda

# Prueba nula (ver Validación): debería dar 0 candidatos
python buscar_masivo.py --sector 20 --max 100 --tmag 9 12 --sin-conocidos --salida nula_s20 --invertir
```

| Opción | Qué hace |
|---|---|
| `--sector N` / `--lista archivo` | De dónde sacar las estrellas (listas oficiales de TESS por sector, o la tuya) |
| `--max N`, `--tmag MIN MAX`, `--semilla S` | Cuántas, qué brillo, y qué muestra al azar (misma semilla = misma lista) |
| `--sin-conocidos` | Salta estrellas que ya tienen TOI, CTOI o planeta confirmado |
| `--sectores K` | Sectores por estrella (1 por defecto; más sectores = períodos más largos, pero más lento) |
| `--procesos P` | Estrellas en paralelo (1 por defecto; 2-4 si tu PC tiene núcleos de sobra) |
| `--tiempo-max S` | Segundos máximos por estrella (900); si se cuelga, se salta |
| `--conservar-datos` | No borrar las curvas de luz descargadas (por defecto se borran, para no llenar el disco) |
| `--actualizar-catalogos` | Bajar versiones frescas de los catálogos antes de empezar |
| `--reintentar-errores` | Volver a intentar las estrellas que fallaron (por ejemplo, por internet) |
| `--solo-resumen` | Regenerar `resumen.md` y `resultados.csv` sin analizar nada |

**Qué deja en la carpeta de salida:**
- `resumen.md`: los candidatos **sin registro** ordenados por probabilidad de IA, con sus advertencias, y cómo revisarlos.
- `resultados.csv`: todas las señales (ábrelo en Excel).
- `graficos/`: un gráfico por cada estrella con candidato.
- `progreso.jsonl`, `lista_objetivos.txt`, `log.txt`: el avance guardado y el registro.

**Cruce con catálogos:** cada candidato se compara (período ±0,5 %, incluidos múltiplos ×2, ×3, ×½ y ×⅓) con los candidatos TESS (TOI), los de la comunidad (CTOI) y los planetas confirmados del NASA Exoplanet Archive. Sale como `CONOCIDO`, `PERÍODO NUEVO` (la estrella tiene otros objetos) o `NUEVO`.

**Velocidad medida:** unos 25 s por estrella con 1 sector, y ~12 s por estrella con `--procesos 2`. 100 estrellas ≈ 20-40 minutos. Se necesita internet todo el tiempo.

**Qué esperar:** en 120 estrellas reales sin objetos conocidos salió 1 candidato débil. Encontrar algo nuevo y real requiere analizar **miles** de estrellas y revisar a mano cada candidato. "NUEVO" significa "no está en estos catálogos", no "planeta descubierto". Si alguno sobrevive a la revisión (pasos en `resumen.md`), el camino oficial es reportarlo como CTOI en ExoFOP.

## Modelo de IA (carpeta `ml/`)

Aprende de los **2.831 candidatos de TESS que la NASA ya resolvió** (catálogo TOI de ExoFOP, octubre 2026): 1.416 planetas (CP/KP) y 1.415 falsos positivos (FP/FA). Es un modelo de *gradient boosting* (`HistGradientBoostingClassifier`) con probabilidades calibradas. **No cambia el veredicto** de las pruebas físicas: agrega una probabilidad, útil para ordenar candidatos.

| Modelo | Cuándo se usa | Variables | AUC por estrella | AUC temporal |
|---|---|---|---|---|
| Completo | se conocen radio, Teff y log g de la estrella | período, duración, profundidad, radio del planeta, datos de la estrella, duración relativa a Kepler | **0,90** | 0,85 |
| Básico | estrella sin datos | período, duración, profundidad | 0,75 | 0,70 |

**Calibración (modelo completo):** cuando dice 0-20 % eran planetas el 6 %; 40-60 % → 48 %; 80-100 % → 89 %.

**Estrellas reales excluidas del entrenamiento:** Pi Mensae c 86 %; TOI-270 b/c/d 99/94/89 %; WASP-18 b 60 %; CM Draconis 2 %. Con las mediciones del buscador da casi lo mismo que con las del catálogo (diferencia ≤ 3 puntos), así que el modelo "entiende" lo que mide el buscador.

**Trampas evitadas:**
- *Fugas de información:* no usa la disposición del pipeline, comentarios ni datos de seguimiento.
- *SNR del catálogo excluido:* sale de todos los sectores observados hasta hoy (Pi Men: 468 contra 24 de una detección), así que no es comparable.
- *"Dato faltante" como pista:* entre los candidatos sin datos de la estrella solo el 7 % son planetas (contra el 55 %), y un modelo único aprendía "faltan datos = falso". Por eso hay dos modelos.
- *Sesgo de selección:* a las estrellas brillantes se les da veredicto el 63 % de las veces y a las débiles el 20 %. Se corrige con pesos.
- *Mismo sistema en entrenamiento y prueba:* validación agrupada por estrella.

**Límites del modelo:**
- Rinde peor con lo nuevo (AUC temporal 0,85 contra 0,90): los candidatos recientes son de estrellas más débiles.
- Más débil en períodos > 10 d (AUC 0,84) y planetas < 4 R⊕ (0,86).
- **No distingue binarias gemelas rasantes** (en simulaciones, mediana 52 % contra 63 % de planetas). No ve la forma del tránsito, que es lo que las delata.
- **Castiga lo raro:** planetas en zonas poco pobladas (ej. tamaño intermedio con período corto, el "desierto de Neptunos") reciben probabilidades bajas aunque sean reales.
- El modelo básico subestima planetas chicos y sobrestima grandes, porque no conoce su tamaño.
- Las etiquetas de la NASA también pueden tener errores.

Reentrenar con el catálogo actualizado:
```bash
cd ml
curl -o datos/toi_exofop.csv "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
python entrenar_clasificador.py
```

## Errores corregidos durante el desarrollo

| # | Error / sesgo | Corrección |
|---|---|---|
| 1 | Siempre reportaba un "candidato", incluso en ruido puro | Umbrales SDE + SNR calibrados con simulaciones |
| 2 | Ventana de limpieza en puntos fijos (19 días con datos de 30 min) | Ventana en días, elegida según la estrella |
| 3 | Grilla de períodos uniforme en período (gruesa en períodos cortos) | Grilla uniforme en frecuencia |
| 4 | Período máximo de 15 d (un planeta de 20 d salía con 10 d) | Hasta 3/4 de la observación, y advierte la ambigüedad |
| 5 | Encontraba un solo planeta | Búsqueda iterativa |
| 6 | No distinguía binarias | 10 pruebas de validación |
| 7 | Probaba "tránsitos" de 10 h en órbitas de 12 h | Solo duraciones físicamente posibles |
| 8 | El filtro se "comía" parte del tránsito | Biweight robusto + máscaras |
| 9 | Error subestimado (filtro asimétrico, ruido rojo) | Error medido empíricamente en momentos al azar |
| 10 | Perdía planetas en estrellas que rotan rápido | Pre-blanqueo con Lomb-Scargle |
| 11 | Elegía 2P en vez de P en señales fuertes | Corrección de armónicos por potencia |
| 12 | Descartaba Júpiter calientes reales por su eclipse secundario | Excepción física (secundario chico y en forma de U) |
| 13 | Radio sobreestimado ~8 % | Corrección por oscurecimiento al limbo |
| 14 | Fallaba con los datos enmascarados de TESS reales | Conversión a arreglos simples |
| 15 | Ordenaba los sectores como texto ("10" antes que "2") | Orden numérico |
| 16 | En datos reales daba 2,5 % de falsas alarmas (prueba nula invertida) | Mínimo 3 tránsitos, prueba de bordes de datos, prueba de duración corta → 0 en 120 |
| 17 | Descartaba planetas reales por fluctuaciones de 3σ en alguna prueba | Descartar recién sobre 4-5σ; entre 3σ y eso, advertencia |
| 18 | Algunos sectores de QLP traen puntos con luz casi en cero, que aparecían como "tránsitos" del 80-99 % | Se descartan puntos físicamente imposibles (bajo 30 % o sobre 200 % de la luz normal) y se avisa cuántos |

## Límites que no se pueden eliminar (solo declarar)

- **Geometría:** solo se ven planetas cuya órbita pasa justo frente a la estrella.
- **Tamaño y período:** favorece planetas grandes y de período corto. Con SNR < 10, la detección baja (ver tabla).
- **Tres tránsitos mínimos:** con 1 sector (~27 días) solo se confirman períodos de hasta ~9-13 días. Para períodos más largos hay que usar más sectores (`--sectores 0`).
- **Binarias de fondo:** otra estrella en el mismo píxel puede imitar un planeta perfecto. La curva de luz sola no lo descarta.
- **Gemelas rasantes:** dos estrellas iguales rozándose se ven como un planeta a la mitad del período. El script lo advierte por la forma en V, pero no puede descartarlo, porque un planeta que pasa rozando el borde de la estrella se ve igual.
- **Variabilidad extrema** (> 10 %): el método es poco confiable, y lo advierte.
- **"Candidato" no es "planeta":** la confirmación requiere otras observaciones (velocidad radial, imágenes de alta resolución).

## Archivos

- `buscador_transitos.py`: el programa.
- `pruebas_validacion.py`: batería estadística con simulaciones (~30 min).
- `validar_reales.py`: validación con estrellas reales de TESS.
- `buscar_masivo.py`: búsqueda masiva con avance guardado y cruce con catálogos.
- `ml/datos/ctoi_exofop.csv`, `ml/datos/confirmados.csv`: catálogos para el cruce (octubre 2026).
- `docs/img/`: gráficos de las estrellas reales y de la validación.
- `ml/caracteristicas.py`: variables, calculadas igual para el catálogo y para el buscador.
- `ml/entrenar_clasificador.py`: entrenamiento y validación (imprime todas las tablas).
- `ml/clasificador_toi.joblib`: modelos entrenados. `ml/informe_clasificador.json`: métricas completas.
- `ml/probar_reales.py` y `ml/probar_simulados.py`: pruebas del modelo.
- `ml/datos/toi_exofop.csv`: copia del catálogo usado (octubre 2026).
- `pruebas_validacion.py` y `validar_reales.py` están en la raíz porque importan `buscador_transitos.py` directamente.
