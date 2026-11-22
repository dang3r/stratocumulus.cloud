import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def fit_head(frame, features, name, regularization, iterations, overrides):
    train = frame.split == "train"
    validation = frame.split == "val"
    counts = frame[train].groupby(["source", "label"]).file.transform("count")
    domains = [
        validation & (frame.source == source) for source in ("CCSN", "Howard", "GLOBE")
    ]
    domain_weights = [1, 1, 2]
    if name == "augmented":
        domains.append(validation & (frame.source == "Facade"))
        domain_weights.append(1)
    background = validation & (frame.source == "ADE")
    best_value = -np.inf
    for exponent in (0.4, 0.7, 1.0):
        weights = counts.astype(float) ** -exponent
        weights /= weights.mean()
        for strength in regularization:
            head = LogisticRegression(C=strength, max_iter=iterations)
            head.fit(features[train], frame.label[train], sample_weight=weights)
            probabilities = head.predict_proba(features)
            sc = probabilities[:, 1]
            cloud_scene = probabilities.argmax(axis=1) != 2
            for override in overrides:
                usable = frame.usable | (sc >= override)
                for threshold in np.arange(0.2, 0.81, 0.025):
                    predictions = (sc >= threshold) & cloud_scene & usable
                    scores = [
                        f1_score(frame.label[part] == 1, predictions[part])
                        for part in domains
                    ]
                    value = (
                        np.average(scores, weights=domain_weights)
                        - predictions[background].mean()
                    )
                    if value > best_value:
                        best_value = value
                        parameters = {
                            "weights": head.coef_.tolist(),
                            "bias": head.intercept_.tolist(),
                            "threshold": threshold,
                            "override": override,
                        }
                        best_probabilities = probabilities
                        best_predictions = predictions
    frame[["other", "sc", "no_sky"]] = best_probabilities
    frame["positive"] = best_predictions
    frame.to_csv(name + "_scores.csv", index=False)
    Path(name + "_head.json").write_text(json.dumps(parameters))
    print("Validation objective", best_value)
