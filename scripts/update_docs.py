"""Fill every results block in README.md, docs/*.md and docs/index.html from results/*.json.

Blocks are delimited by ``<!-- BEGIN:name -->`` / ``<!-- END:name -->``. No metric in the docs is
typed by hand: re-running this script after ``make train export benchmark`` regenerates them all.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ecg_edge import SCORED_CLASSES
from ecg_edge.utils import PROJECT_ROOT, RESULTS_DIR, get_logger, load_json

log = get_logger("update_docs")
VARIANTS = ("tiny", "small", "base")
TARGETS = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "REPORT.md",
    PROJECT_ROOT / "docs" / "DECISIONS.md",
    PROJECT_ROOT / "docs" / "index.html",
    PROJECT_ROOT / "HANDOFF.md",
]

Table = tuple[list[str], list[list[str]]]


class Ctx:
    """Lazily loaded result files."""

    def __init__(self) -> None:
        self.bench = load_json(RESULTS_DIR / "benchmark.json")
        self.export = load_json(RESULTS_DIR / "export.json")
        self.card = load_json(RESULTS_DIR / "data_card.json")
        self.best = load_json(RESULTS_DIR / "best_model.json")
        self.sweep = load_json(RESULTS_DIR / "sweep_tiny.json")
        self.train = {v: load_json(RESULTS_DIR / f"train_inter_{v}.json") for v in VARIANTS}
        bv = self.best["variant"]
        self.inter_eval = load_json(RESULTS_DIR / f"eval_inter_{bv}.json")["test"]
        intra_p = RESULTS_DIR / f"eval_intra_{bv}.json"
        self.intra_eval = load_json(intra_p)["test"] if intra_p.exists() else None

    def row(self, variant: str, runtime: str) -> dict[str, Any]:
        return next(r for r in self.bench["rows"] if r["variant"] == variant and r["runtime"] == runtime)


def f3(x: float) -> str:
    return f"{x:.3f}"


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


# ---------------------------------------------------------------- blocks
def results_table(c: Ctx) -> Table:
    head = ["Variant", "Runtime", "Params", "Size (KB)", "Accuracy", "Macro-F1", *[f"F1 {k}" for k in SCORED_CLASSES]]
    head += ["p50 (ms)", "p95 (ms)", "Beats/s"]
    rows = []
    for r in c.bench["rows"]:
        name = f"**{r['variant']}**" if r["variant"] == c.bench["best_variant"] else r["variant"]
        rows.append(
            [name, r["runtime_label"], f"{r['params']:,}", f"{r['size_kb']:.1f}", f3(r["accuracy"]), f3(r["macro_f1"])]
            + [f3(r["per_class_f1"][k]) for k in SCORED_CLASSES]
            + [f"{r['latency'][k]:.3f}" for k in ("p50_ms", "p95_ms")]
            + [f"{r['latency']['throughput_beats_per_s']:,.0f}"]
        )  # fmt: skip
    return head, rows


def headline_table(c: Ctx) -> Table:
    bv = c.bench["best_variant"]
    head = [
        "Model",
        "Size (KB)",
        "Accuracy",
        "Macro-F1 (N,S,V,F)",
        *[f"F1 {k}" for k in SCORED_CLASSES],
        "p50 latency (ms)",
    ]
    rows = []
    for rt in ("pytorch_fp32", "litert_fp32", "litert_int8"):
        r = c.row(bv, rt)
        rows.append(
            [f"{bv} - {r['runtime_label']}", f"{r['size_kb']:.1f}", f3(r["accuracy"]), f3(r["macro_f1"])]
            + [f3(r["per_class_f1"][k]) for k in SCORED_CLASSES]
            + [f"{r['latency']['p50_ms']:.3f}"]
        )
    return head, rows


def per_class_table(c: Ctx) -> Table:
    bv = c.bench["best_variant"]
    m = c.export[bv]["litert_int8"]["test"]["per_class"]
    head = ["Class", "Support", "Precision (PPV)", "Recall (Se)", "F1"]
    rows = [[k, f"{m[k]['support']:,}", f3(m[k]["precision"]), f3(m[k]["recall"]), f3(m[k]["f1"])] for k in m]
    return head, rows


def quant_table(c: Ctx) -> Table:
    head = [
        "Variant",
        "FP32 -> int8 size",
        "Macro-F1 delta",
        "Accuracy delta",
        "Speed-up vs LiteRT FP32",
        "Speed-up vs PyTorch",
    ]
    head += ["FP32 .tflite agreement w/ PyTorch", "int8 agreement w/ PyTorch"]
    rows = []
    for v in VARIANTS:
        q = c.bench["quantization_summary"][v]
        rows.append([
            v,
            f"{q['size_reduction_x']:.2f}x smaller",
            f"{q['macro_f1_delta']:+.4f}",
            f"{q['accuracy_delta']:+.4f}",
            f"{q['speedup_vs_litert_fp32_p50']:.2f}x",
            f"{q['speedup_vs_pytorch_fp32_p50']:.1f}x",
            pct(c.export[v]["litert_fp32"]["agreement_with_pytorch"]),
            pct(c.export[v]["litert_int8"]["agreement_with_pytorch"]),
        ])  # fmt: skip
    return head, rows


def protocol_table(c: Ctx) -> Table:
    head = ["Protocol (same architecture, PyTorch FP32)", "Test beats", "Accuracy", "Macro-F1 (N,S,V,F)"]
    head += [f"F1 {k}" for k in SCORED_CLASSES]
    rows = []
    items = [("**Inter-patient** (DS1 -> DS2, unseen patients)", c.inter_eval)]
    if c.intra_eval:
        items.append(("Intra-patient random 80/20 split (leaky, optimistic)", c.intra_eval))
    for name, m in items:
        rows.append(
            [name, f"{m['n']:,}", f3(m["accuracy"]), f3(m["macro_f1"])]
            + [f3(m["per_class"][k]["f1"]) for k in SCORED_CLASSES]
        )
    return head, rows


def data_table(c: Ctx) -> Table:
    classes = ["N", "S", "V", "F", "Q"]
    head = ["Split", "Records", "Beats", *classes]
    rows = []
    for split, info in c.card["inter_patient"].items():
        rows.append(
            [split, str(len(info["records"])), f"{info['n_beats']:,}"]
            + [f"{info['class_counts'][k]:,}" for k in classes]
        )
    return head, rows


def training_table(c: Ctx) -> Table:
    head = [
        "Variant",
        "Params",
        "Best epoch",
        "Epochs run",
        "Train time (s, CPU)",
        "Val F1 (N,S,V)",
        "Val macro-F1 (N,S,V,F)",
    ]
    rows = []
    for v in VARIANTS:
        t = c.train[v]
        rows.append([
            f"**{v}**" if v == c.bench["best_variant"] else v,
            f"{t['params']:,}",
            str(t["best_epoch"]),
            str(t["epochs_run"]),
            f"{t['train_seconds']:.0f}",
            f3(t["val_select_f1"]),
            f3(t["val"]["macro_f1"]),
        ])  # fmt: skip
    return head, rows


def sweep_table(c: Ctx) -> Table:
    head = ["Run", "Overrides vs. base", "Best epoch", "Val F1 (N,S,V)", "Val F1 S", "Val F1 V"]
    rows = []
    for r in c.sweep["runs"]:
        ov = ", ".join(f"{k}={v}" for k, v in r["overrides"].items()) or "-"
        rows.append(
            [r["run"], ov, str(r["best_epoch"]), f3(r["val_select_f1_NSV"]), f3(r["val_f1"]["S"]), f3(r["val_f1"]["V"])]
        )
    return head, rows


def headline_text(c: Ctx) -> str:
    h = c.bench["headline"]
    q = c.bench["quantization_summary"][h["variant"]]
    test = c.card["inter_patient"]["test"]
    return (
        f"Best model (**{h['variant']}**, chosen on validation): full-integer **int8 LiteRT, {h['size_kb']:.1f} KB**, "
        f"**macro-F1 {h['macro_f1']:.3f}** (N,S,V,F) and accuracy {h['accuracy']:.3f} on "
        f"{test['n_beats']:,} beats from {len(test['records'])} *unseen* patients (DS2); "
        f"**{q['size_reduction_x']:.1f}x smaller** than its FP32 .tflite with a macro-F1 "
        f"change of {q['macro_f1_delta']:+.4f}; p50 single-beat latency **{h['p50_ms']:.3f} ms** "
        f"({c.bench['hardware']['label']})."
    )


def hardware_text(c: Ctx) -> str:
    hw = c.bench["hardware"]
    s = c.bench["latency_settings"]
    return (
        f"Latency measured on **{hw['cpu']}** ({hw['arch']}, {hw['os']}{', WSL2' if hw['wsl2'] else ''}), "
        f"Python {hw['python']}, **{s['threads']} thread, batch {s['batch']}**, {s['warmup']} warm-up + "
        f"{s['runs']} timed runs. This is **desktop x86 CPU latency, not phone latency** - see "
        f"[`scripts/android_benchmark.md`](scripts/android_benchmark.md) for on-device measurement."
    )


def quant_text(c: Ctx) -> str:
    bv = c.bench["best_variant"]
    q = c.bench["quantization_summary"][bv]
    fp32_agree = min(c.export[v]["litert_fp32"]["agreement_with_pytorch"] for v in VARIANTS)
    sizes = [c.bench["quantization_summary"][v]["size_reduction_x"] for v in VARIANTS]
    deltas = [c.bench["quantization_summary"][v]["macro_f1_delta"] for v in VARIANTS]
    i8 = [c.export[v]["litert_int8"]["agreement_with_pytorch"] for v in VARIANTS]
    return (
        f"- **Size:** int8 files are {min(sizes):.1f}-{max(sizes):.1f}x smaller than FP32 .tflite "
        f"({bv}: {c.row(bv, 'litert_fp32')['size_kb']:.1f} KB -> {c.row(bv, 'litert_int8')['size_kb']:.1f} KB).\n"
        f"- **Accuracy:** macro-F1 change after int8 PTQ ranges {min(deltas):+.4f} to {max(deltas):+.4f} "
        f"across variants ({bv}: {q['macro_f1_delta']:+.4f}).\n"
        f"- **Speed (host CPU):** {bv} int8 is {q['speedup_vs_litert_fp32_p50']:.2f}x the speed of LiteRT FP32 and "
        f"{q['speedup_vs_pytorch_fp32_p50']:.1f}x the speed of eager PyTorch at batch 1.\n"
        f"- **Parity:** FP32 .tflite predictions agree with PyTorch on {pct(fp32_agree)} of DS2 beats (minimum over "
        f"variants; requirement > 99.9%); full-integer int8 agreement is {pct(min(i8))}-{pct(max(i8))}."
    )


def protocol_text(c: Ctx) -> str:
    if not c.intra_eval:
        return "_Intra-patient run not available._"
    gap = c.intra_eval["macro_f1"] - c.inter_eval["macro_f1"]
    return (
        f"The identical architecture scores macro-F1 **{c.intra_eval['macro_f1']:.3f}** on a random beat-level split "
        f"but **{c.inter_eval['macro_f1']:.3f}** on unseen patients - a gap of {gap:.3f}. The random split lets "
        f"the network memorise each patient's beat morphology, because neighbouring beats of the same person sit in "
        f"both train and test."
    )


TABLES: dict[str, Callable[[Ctx], Table]] = {
    "results_table": results_table,
    "headline_table": headline_table,
    "per_class_table": per_class_table,
    "quant_table": quant_table,
    "protocol_table": protocol_table,
    "data_table": data_table,
    "training_table": training_table,
    "sweep_table": sweep_table,
}
TEXTS: dict[str, Callable[[Ctx], str]] = {
    "headline": headline_text,
    "hardware": hardware_text,
    "quant_findings": quant_text,
    "protocol_text": protocol_text,
}


# ---------------------------------------------------------------- rendering
NUMERIC = re.compile(r"^[+-]?[\d,.]+(%|x|x smaller)?$")


def md_table(t: Table) -> str:
    head, rows = t
    align = ["---:" if all(NUMERIC.match(r[i]) for r in rows) else "---" for i in range(len(head))]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def inline_md_to_html(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
    s = re.sub(
        r"\[`(.+?)`\]\((.+?)\)",
        r'<a href="https://github.com/nikhil3495/ECG-Arrhythmia-Classifier/blob/main/\2"><code>\1</code></a>',
        s,
    )
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def html_table(t: Table) -> str:
    head, rows = t
    th = "".join(f"<th>{inline_md_to_html(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{inline_md_to_html(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def html_text(s: str) -> str:
    lines = s.split("\n")
    if all(line.startswith("- ") for line in lines):
        return "<ul>" + "".join(f"<li>{inline_md_to_html(line[2:])}</li>" for line in lines) + "</ul>"
    return f"<p>{inline_md_to_html(s)}</p>"


def render(name: str, ctx: Ctx, is_html: bool) -> str:
    if name in TABLES:
        t = TABLES[name](ctx)
        return html_table(t) if is_html else md_table(t)
    s = TEXTS[name](ctx)
    return html_text(s) if is_html else s


BLOCK = re.compile(r"(<!-- BEGIN:(\w+) -->)(.*?)(<!-- END:\2 -->)", re.S)


def fill(path: Path, ctx: Ctx) -> int:
    text = path.read_text(encoding="utf-8")
    is_html = path.suffix == ".html"
    n = 0

    def sub(m: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return f"{m.group(1)}\n{render(m.group(2), ctx, is_html)}\n{m.group(4)}"

    new = BLOCK.sub(sub, text)
    path.write_text(new, encoding="utf-8")
    return n


def main() -> None:
    ctx = Ctx()
    for path in TARGETS:
        if path.exists():
            log.info("%s: filled %d blocks", path.relative_to(PROJECT_ROOT), fill(path, ctx))


if __name__ == "__main__":
    main()
