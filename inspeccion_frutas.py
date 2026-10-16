"""Clasificación de calidad de frutas (jitomate) con visión por computadora y Random Forest.

Extrae 8 características de una foto de la fruta sobre fondo oscuro o claro,
y un Random Forest la clasifica en Exportacion_Grado_A, Madurez_Media_Grado_B o Rechazo_Defecto_Plaga.

    python inspeccion_frutas.py entrenar                     # con el dataset sintético de prueba
    python inspeccion_frutas.py entrenar --csv datos.csv     # con tus datos (columnas FEATURES + quality_label)
    python inspeccion_frutas.py clasificar foto.jpg [--mqtt 127.0.0.1]
"""
import argparse
import json

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["mean_R", "mean_G", "mean_B", "mean_H", "mean_S", "std_V", "defect_ratio", "circularity"]
CLASSES = ["Exportacion_Grado_A", "Madurez_Media_Grado_B", "Rechazo_Defecto_Plaga"]
MODEL_FILE, SCALER_FILE = "random_forest_fruit_model.pkl", "fruit_feature_scaler.pkl"


# ---------------------------------------------------------------- características
def circular_hue_mean(h: np.ndarray) -> float:
    """Media del tono de OpenCV (0-179 = 0-358°) respetando que el rojo está en ambos extremos.

    Regresa un valor en (-90, 90]: rojo cerca de 0, naranja ~15, amarillo ~30, verde ~60.
    """
    ang = h.astype(np.float64) * (2.0 * np.pi / 180.0)
    m = np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2.0
    return float(m if m > -90.0 else m + 180.0)


def extract_fruit_features(image_bgr: np.ndarray) -> np.ndarray:
    """[mean_R, mean_G, mean_B, mean_H, mean_S, std_V, defect_ratio, circularity]"""
    img = cv2.resize(image_bgr, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Máscara de la fruta: descarta el fondo (poca saturación o muy oscuro)
    mask = cv2.inRange(hsv, np.array([0, 35, 30]), np.array([179, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    total = float(cv2.countNonZero(mask))
    if total < 100.0:
        return np.zeros(8, dtype=np.float32)

    mean_b, mean_g, mean_r, _ = cv2.mean(img, mask=mask)
    inside = mask > 0
    mean_h = circular_hue_mean(hsv[:, :, 0][inside])
    mean_s = float(hsv[:, :, 1][inside].mean())
    std_v = float(np.std(hsv[:, :, 2][inside]))

    # Manchas oscuras o lesiones dentro de la fruta
    defect = cv2.bitwise_and(cv2.inRange(hsv, np.array([0, 40, 10]), np.array([179, 255, 85])), mask)
    defect_ratio = float(cv2.countNonZero(defect)) / total

    # Circularidad 4πA/P² del contorno principal (1 = círculo perfecto)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        area, perim = cv2.contourArea(c), cv2.arcLength(c, True)
        if perim > 0:
            circularity = 4.0 * np.pi * area / perim ** 2

    return np.array([mean_r, mean_g, mean_b, mean_h, mean_s, std_v, defect_ratio, circularity], dtype=np.float32)


# ---------------------------------------------------------------- datos
def synthetic_dataset(n_per_class: int = 300, seed: int = 42):
    """Dataset sintético para probar el flujo completo. No sustituye fotos reales."""
    rng = np.random.default_rng(seed)
    n = n_per_class
    c0 = np.column_stack([rng.normal(210, 12, n), rng.normal(45, 10, n), rng.normal(40, 10, n),
                          rng.normal(6, 3, n), rng.normal(210, 15, n), rng.normal(18, 4, n),
                          rng.uniform(0.00, 0.03, n), rng.uniform(0.84, 0.96, n)])
    c1 = np.column_stack([rng.normal(190, 15, n), rng.normal(135, 18, n), rng.normal(50, 12, n),
                          rng.normal(22, 5, n), rng.normal(175, 20, n), rng.normal(26, 5, n),
                          rng.uniform(0.02, 0.08, n), rng.uniform(0.74, 0.88, n)])
    c2 = np.column_stack([rng.normal(125, 22, n), rng.normal(95, 20, n), rng.normal(70, 18, n),
                          rng.normal(35, 12, n), rng.normal(120, 25, n), rng.normal(48, 9, n),
                          rng.uniform(0.10, 0.38, n), rng.uniform(0.45, 0.75, n)])
    X = np.vstack([c0, c1, c2])
    y = np.array([CLASSES[0]] * n + [CLASSES[1]] * n + [CLASSES[2]] * n)
    return X, y


# ---------------------------------------------------------------- entrenamiento
def train_fruit_random_forest(csv_path: str | None = None, save: bool = True, verbose: bool = True):
    if csv_path:
        df = pd.read_csv(csv_path)
        X, y = df[FEATURES].values, df["quality_label"].values
    else:
        X, y = synthetic_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    rf = RandomForestClassifier(n_estimators=200, criterion="gini", max_depth=12, min_samples_split=4,
                                min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(scaler.transform(X_tr), y_tr)
    y_pred = rf.predict(scaler.transform(X_te))
    acc = accuracy_score(y_te, y_pred)
    if verbose:
        print(f"Exactitud en el conjunto de prueba: {acc * 100:.2f} %" + ("" if csv_path else " (dataset sintético)"))
        print(classification_report(y_te, y_pred))
        for name, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda t: -t[1]):
            print(f"  {name:13s} {imp:.4f}")
    if save:
        joblib.dump(rf, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
    return rf, scaler, acc


# ---------------------------------------------------------------- inferencia
def classify_fruit_sample(features: np.ndarray, rf, scaler, batch_code: str = "LOTE-IBERO-2025") -> dict:
    Xs = scaler.transform(np.asarray(features, dtype=np.float64).reshape(1, -1))
    label = rf.predict(Xs)[0]
    conf = float(np.max(rf.predict_proba(Xs)[0])) * 100.0
    return {
        "batch_code": batch_code,
        "predicted_class": str(label),
        "confidence_pct": round(conf, 2),
        "defect_ratio": round(float(features[6]), 4),
        "circularity": round(float(features[7]), 4),
        "mean_rgb": [round(float(v), 1) for v in features[:3]],
    }


def publish_mqtt(payload: dict, broker: str = "127.0.0.1", topic: str = "fruits/inspection/result") -> bool:
    """Publica el dictamen. Regresa False si no hay broker; no detiene la inspección."""
    try:
        import paho.mqtt.client as mqtt
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect(broker, 1883, 60)
        client.publish(topic, json.dumps(payload))
        client.disconnect()
        return True
    except Exception as e:  # sin broker o sin red
        print(f"[AVISO MQTT] No se publicó en {broker}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("entrenar"); t.add_argument("--csv")
    c = sub.add_parser("clasificar"); c.add_argument("imagen"); c.add_argument("--mqtt")
    a = ap.parse_args()
    if a.cmd == "entrenar":
        train_fruit_random_forest(a.csv)
    else:
        img = cv2.imread(a.imagen)
        if img is None:
            raise SystemExit(f"No se pudo leer {a.imagen}")
        rf, sc = joblib.load(MODEL_FILE), joblib.load(SCALER_FILE)
        res = classify_fruit_sample(extract_fruit_features(img), rf, sc)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        if a.mqtt:
            publish_mqtt(res, a.mqtt)


if __name__ == "__main__":
    main()
