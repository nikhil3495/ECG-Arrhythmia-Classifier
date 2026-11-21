"""Pick the best inter-patient variant by *validation* macro-F1 (never by test) -> results/best_model.json."""

from __future__ import annotations

import argparse

from ecg_edge.utils import RESULTS_DIR, get_logger, load_json, save_json

log = get_logger("select_best")
VARIANTS = ("tiny", "small", "base")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-config", action="store_true", help="print the winning config path and exit")
    args = parser.parse_args()
    scores = {v: load_json(RESULTS_DIR / f"train_inter_{v}.json")["val"]["macro_f1"] for v in VARIANTS}
    best = max(scores, key=lambda v: scores[v])
    save_json(
        {
            "variant": best,
            "config": f"configs/{best}.yaml",
            "selection": "highest validation macro-F1 (N,S,V,F) on the DS1 patient-wise hold-out",
            "val_macro_f1": scores,
        },
        RESULTS_DIR / "best_model.json",
    )
    if args.print_config:
        print(f"configs/{best}.yaml")
    else:
        log.info("best variant by val macro-F1: %s %s", best, scores)


if __name__ == "__main__":
    main()
