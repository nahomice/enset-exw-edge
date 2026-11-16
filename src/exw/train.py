"""FP32 training of the tile CNN (Stage 3 of the workflow)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from exw.constants import N_CLASSES
from exw.data.dataset import class_counts
from exw.models.tilecnn import build_tilecnn, count_params, preprocess


def class_weights(y: np.ndarray) -> dict[int, float]:
    """Inverse-frequency weights (background tiles dominate)."""
    counts = np.array(class_counts(y, N_CLASSES), float)
    w = counts.sum() / (N_CLASSES * np.maximum(counts, 1))
    return {i: float(w[i]) for i in range(N_CLASSES)}


def _augment(x, y, w):
    """Training-only augmentation. Colour shifts are kept small: yellowing IS the symptom."""
    x = tf.image.random_flip_left_right(x)
    x = x + tf.random.normal(tf.shape(x), 0.0, 0.02)
    x = tf.image.random_brightness(x, 0.06)
    x = tf.image.random_contrast(x, 0.9, 1.1)
    return tf.clip_by_value(x, -1.0, 0.9922), y, w


def make_ds(X_uint8, y, weights: dict, batch: int, train: bool, seed: int):
    sw = np.array([weights[int(c)] for c in y], np.float32)
    ds = tf.data.Dataset.from_tensor_slices((preprocess(X_uint8), y.astype(np.int32), sw))
    if train:
        ds = ds.shuffle(min(len(y), 20000), seed=seed, reshuffle_each_iteration=True).map(
            _augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def train_one(cfg: dict, tr: dict, va: dict, seed: int, out_dir: str | Path) -> dict:
    """Train one seed; saves ``model_seed{seed}.keras`` and ``history_seed{seed}.json``."""
    tc = cfg["train"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(seed)
    model = build_tilecnn(cfg["tiles"]["size"], cfg["model"]["width"])
    n_params = count_params(model)
    assert n_params < cfg["model"]["max_params"], f"{n_params} parameters exceed the budget"
    cw = class_weights(tr["y"])
    model.compile(optimizer=tf.keras.optimizers.Adam(tc["lr"]),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    hist = model.fit(
        make_ds(tr["X"], tr["y"], cw, tc["batch_size"], True, seed),
        validation_data=make_ds(va["X"], va["y"], {i: 1.0 for i in range(N_CLASSES)}, tc["batch_size"], False, seed),
        epochs=tc["epochs"], verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=tc["patience"], restore_best_weights=True)])
    model.save(out_dir / f"model_seed{seed}.keras")
    info = dict(seed=seed, n_params=n_params, class_weights=cw,
                epochs_run=len(hist.history["loss"]),
                val_loss=[float(v) for v in hist.history["val_loss"]],
                train_loss=[float(v) for v in hist.history["loss"]],
                val_acc=[float(v) for v in hist.history["val_accuracy"]])
    with open(out_dir / f"history_seed{seed}.json", "w") as f:
        json.dump(info, f, indent=1)
    return info
