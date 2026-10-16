import pathlib, sys
import cv2, numpy as np, pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import inspeccion_frutas as F


def fruit(hue, spread=6, spots=0, shape="circle", seed=0):
    """Jitomate sintético sobre fondo negro (tono de OpenCV 0-179)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:256, :256]
    m = ((xx - 128) ** 2 + (yy - 128) ** 2 < 90 ** 2) if shape == "circle" else (((xx - 128) / 110) ** 2 + ((yy - 128) / 60) ** 2 < 1)
    hsv = np.zeros((256, 256, 3), np.uint8)
    hsv[..., 0] = (hue + rng.integers(-spread, spread + 1, (256, 256))) % 180
    hsv[..., 1], hsv[..., 2] = 210, 200
    for _ in range(spots):
        cx, cy = rng.integers(90, 166, 2)
        hsv[..., 2][((xx - cx) ** 2 + (yy - cy) ** 2 < 14 ** 2) & m] = 60
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR); img[~m] = 0
    return img


def test_red_hue_crossing_zero_is_near_zero():
    # el rojo cae en ambos extremos (0 y 179); una media normal daría ~84
    h = F.extract_fruit_features(fruit(0))[3]
    assert abs(h) < 1.0


@pytest.mark.parametrize("hue,expected", [(3, 3), (15, 15), (30, 30), (60, 60)])
def test_hue_follows_color(hue, expected):
    assert F.extract_fruit_features(fruit(hue, 2))[3] == pytest.approx(expected, abs=0.6)


def test_defects_and_shape():
    clean, spotted, oval = (F.extract_fruit_features(x) for x in (fruit(2, 2), fruit(2, 2, spots=6), fruit(2, 2, shape="ellipse")))
    assert clean[6] == 0.0 and spotted[6] > 0.08          # proporción de manchas
    assert spotted[5] > clean[5] + 20                     # más variación de brillo
    assert clean[7] > 0.85 and oval[7] < clean[7] - 0.08  # circularidad


def test_empty_image_gives_zeros():
    assert not F.extract_fruit_features(np.zeros((200, 200, 3), np.uint8)).any()


def test_training_and_inference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rf, sc, acc = F.train_fruit_random_forest(save=True, verbose=False)
    assert acc > 0.95                                      # dataset sintético: clases bien separadas
    assert (tmp_path / F.MODEL_FILE).exists() and (tmp_path / F.SCALER_FILE).exists()
    res = F.classify_fruit_sample(np.array([208, 42, 38, 5.5, 215, 16.2, 0.012, 0.91]), rf, sc)
    assert res["predicted_class"] == "Exportacion_Grado_A" and res["confidence_pct"] > 80
    bad = F.classify_fruit_sample(np.array([120, 95, 70, 35, 120, 50, 0.3, 0.55]), rf, sc)
    assert bad["predicted_class"] == "Rechazo_Defecto_Plaga"


def test_csv_training(tmp_path):
    import pandas as pd
    X, y = F.synthetic_dataset(60, seed=1)
    df = pd.DataFrame(X, columns=F.FEATURES); df["quality_label"] = y
    p = tmp_path / "datos.csv"; df.to_csv(p, index=False)
    _, _, acc = F.train_fruit_random_forest(str(p), save=False, verbose=False)
    assert acc > 0.9


def test_mqtt_without_broker_does_not_raise():
    assert F.publish_mqtt({"x": 1}, broker="127.0.0.1") is False
