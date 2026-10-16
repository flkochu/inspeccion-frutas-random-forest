# Inspección de frutas con Random Forest

Clasifica la calidad de un jitomate a partir de una foto: extrae 8 características de color, defectos y forma con OpenCV, y un Random Forest de scikit-learn decide entre tres clases.

| Clase | Qué representa |
|---|---|
| `Exportacion_Grado_A` | Rojo intenso, casi sin manchas, redondo |
| `Madurez_Media_Grado_B` | Naranja o amarillo, manchas leves |
| `Rechazo_Defecto_Plaga` | Manchas oscuras o lesiones, forma irregular |

**Autor:** Adán López Lozano

## Características

| # | Nombre | Cómo se calcula |
|---|---|---|
| 1–3 | `mean_R`, `mean_G`, `mean_B` | Promedio de color dentro de la máscara de la fruta |
| 4 | `mean_H` | Tono promedio circular: rojo ≈ 0, naranja ≈ 15, amarillo ≈ 30, verde ≈ 60 |
| 5 | `mean_S` | Saturación promedio |
| 6 | `std_V` | Desviación estándar del brillo (las manchas la suben) |
| 7 | `defect_ratio` | Área de manchas oscuras ÷ área de la fruta |
| 8 | `circularity` | 4πA / P² del contorno (1 = círculo) |

El modelo: 200 árboles, criterio de Gini, `max_features = √p`, clases balanceadas. Cada árbol vota y la clase final es la mayoría; la confianza es la fracción de votos.

## Uso

```bash
pip install opencv-python scikit-learn pandas joblib paho-mqtt

python inspeccion_frutas.py entrenar                    # dataset sintético de prueba
python inspeccion_frutas.py entrenar --csv datos.csv    # tus datos
python inspeccion_frutas.py clasificar foto.jpg         # imprime el dictamen en JSON
python inspeccion_frutas.py clasificar foto.jpg --mqtt 192.168.1.10   # y lo publica en fruits/inspection/result
```

El CSV lleva las 8 columnas de características y una columna `quality_label`. Para armarlo con fotos propias, aplica `extract_fruit_features` a cada foto y anota su clase.

## Sobre los datos

El modelo se entrenó con un **dataset sintético**: 900 muestras generadas con distribuciones típicas de cada clase. Sirve para probar el flujo completo, pero su exactitud (100 %) no dice cómo funcionaría con jitomates reales, porque las clases sintéticas están muy separadas. El siguiente paso es entrenarlo con fotos etiquetadas.

## Qué se verificó

```bash
python -m pytest tests     # 10 pruebas
```

- Con jitomates sintéticos dibujados en imágenes, el tono sigue al color (3, 15, 30 y 60), las manchas suben `defect_ratio` y `std_V`, y una forma ovalada baja la circularidad.
- Una imagen vacía da un vector de ceros.
- Entrenamiento con el dataset sintético y con un CSV, guardado del modelo, y clasificación de una muestra buena y una mala.
- Sin broker MQTT, la inspección sigue y solo avisa.

## Cambios respecto a la versión anterior

- **Tono del rojo**: el rojo cae en ambos extremos del tono de OpenCV (0 y 179), y el promedio normal de un jitomate rojo daba ~84, como si fuera verde-azul. Ahora se usa una media circular.
- **paho-mqtt 2.x**: el cliente se crea con `CallbackAPIVersion.VERSION2`; la forma anterior está obsoleta.
- El dataset sintético quedó en su propia función, `synthetic_dataset()`, para que quede claro qué es de prueba.
- Línea de comandos para entrenar y clasificar.
