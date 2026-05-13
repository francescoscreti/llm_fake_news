"""
comparatore: analizza i risultati della pipeline

uso:
    python comparator.py           → analizza outputs_run1.json
    python comparator.py 2         → analizza outputs_run2.json
    python comparator.py --runs 3  → analizza run1, run2, run3 e fa la media
    python comparator.py --runs    → analizza tutte le run disponibili
"""

import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ─── caricamento ──────────────────────────────────────────────────────────────

def count_runs() -> int:
    """conta i file outputs_run*.json disponibili nella cartella results"""
    if not os.path.exists(RESULTS_DIR):
        return 0

    return len(
        [
            f
            for f in os.listdir(RESULTS_DIR)
            if f.startswith("outputs_run") and f.endswith(".json")
        ]
    )


def load_run(run_id: int) -> list:
    """carica i risultati di una singola run dal file JSON"""
    path = os.path.join(RESULTS_DIR, f"outputs_run{run_id}.json")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_runs(n: int) -> list[list]:
    """carica i risultati di n run successive"""
    return [load_run(i + 1) for i in range(n)]


# ─── metriche singola run ─────────────────────────────────────────────────────

def compute_metrics(results: list, strategy: str) -> dict:
    """calcola le metriche di accuratezza per una strategia su una run"""
    total = len(results)

    # conta le predizioni corrette per la strategia specificata
    correct = sum(1 for r in results if r[strategy]["correct"])

    # separa i risultati per classe di verità
    fake = [r for r in results if r["ground_truth"] == "FAKE"]
    real = [r for r in results if r["ground_truth"] == "REAL"]

    # calcola accuratezza per classe
    fake_correct = sum(1 for r in fake if r[strategy]["correct"])
    real_correct = sum(1 for r in real if r[strategy]["correct"])

    # conta le predizioni non parsabili
    unknown = sum(1 for r in results if r[strategy]["label"] == "UNKNOWN")

    return {
        "accuracy": correct / total if total else 0,
        "acc_fake": fake_correct / len(fake) if fake else 0,
        "acc_real": real_correct / len(real) if real else 0,
        "unknown_count": unknown,
        "correct": correct,
        "total": total,
    }


# ─── metriche multi-run ───────────────────────────────────────────────────────

def compute_multi_metrics(runs: list[list], strategy: str) -> dict:
    """calcola le metriche aggregate e la stabilità su più run"""

    # estrae le accuratezze da tutte le run
    accuracies = [compute_metrics(r, strategy)["accuracy"] for r in runs]
    acc_fake = [compute_metrics(r, strategy)["acc_fake"] for r in runs]
    acc_real = [compute_metrics(r, strategy)["acc_real"] for r in runs]

    # funzioni di utilità per il calcolo statistico
    mean = lambda xs: sum(xs) / len(xs) if xs else 0
    stdev = lambda xs: (
        sum((x - mean(xs)) ** 2 for x in xs) / len(xs)
    ) ** 0.5 if xs else 0

    # analizza la stabilità:
    # quante volte ogni sample è stato classificato correttamente
    n_samples = len(runs[0])

    stability = []

    for i in range(n_samples):
        correct_count = sum(
            1 for run in runs if run[i][strategy]["correct"]
        )

        stability.append({
            "id": runs[0][i]["id"],
            "text_preview": runs[0][i]["text_preview"],
            "ground_truth": runs[0][i]["ground_truth"],
            "correct_runs": correct_count,
            "total_runs": len(runs),
        })

    return {
        "mean_accuracy": mean(accuracies),
        "std_accuracy": stdev(accuracies),
        "mean_acc_fake": mean(acc_fake),
        "mean_acc_real": mean(acc_real),
        "per_run": accuracies,
        "stability": stability,
    }


# ─── stampa ───────────────────────────────────────────────────────────────────

def print_metrics(label: str, m: dict):
    """stampa le metriche di una singola run in formato leggibile"""

    print(f"\n{'─' * 40}")
    print(f"  {label}")
    print(f"{'─' * 40}")

    print(f"  accuracy globale : {m['accuracy']:.2f}  ({m['correct']}/{m['total']})")
    print(f"  accuracy FAKE    : {m['acc_fake']:.2f}")
    print(f"  accuracy REAL    : {m['acc_real']:.2f}")

    if m["unknown_count"] > 0:
        print(f"  ⚠ output non parsabili: {m['unknown_count']}")


def print_multi_metrics(label: str, m: dict):
    """stampa le metriche aggregate su più run con media e deviazione standard"""

    print(f"\n{'─' * 40}")
    print(f"  {label}")
    print(f"{'─' * 40}")

    print(f"  accuracy media   : {m['mean_accuracy']:.2f} ± {m['std_accuracy']:.2f}")
    print(f"  accuracy FAKE    : {m['mean_acc_fake']:.2f}")
    print(f"  accuracy REAL    : {m['mean_acc_real']:.2f}")
    print(f"  per run          : {[f'{a:.2f}' for a in m['per_run']]}")


def print_stability(zs: dict, fs: dict):
    """analizza e stampa i sample con comportamento instabile tra le run"""

    print(f"\n{'─' * 40}")
    print("  stabilità per sample")
    print(f"{'─' * 40}")

    # identifica i sample classificati in modo non consistente
    n_runs = zs["stability"][0]["total_runs"]

    unstable = [
        s
        for s in zs["stability"]
        if s["correct_runs"] < n_runs
        or fs["stability"][s["id"]]["correct_runs"] < n_runs
    ]

    if not unstable:
        print("  tutti i sample classificati correttamente in ogni run")
        return

    for s in unstable:
        fs_s = fs["stability"][s["id"]]

        print(f"  [id {s['id']}] GT={s['ground_truth']}")
        print(f"    zero-shot corretto {s['correct_runs']}/{n_runs} run")
        print(f"    few-shot  corretto {fs_s['correct_runs']}/{n_runs} run")
        print(f"    {s['text_preview'][:100]}...")


def print_errors(results: list, strategy: str):
    """stampa i sample classificati erroneamente per una strategia"""

    # filtra i risultati errati
    errors = [r for r in results if not r[strategy]["correct"]]

    if not errors:
        print("  nessun errore")
        return

    for r in errors:
        print(f"  [id {r['id']}] GT={r['ground_truth']} → {r[strategy]['label']}")
        print(f"    {r['text_preview'][:120]}...")


def print_disagreements(results: list):
    """analizza i casi dove zero-shot e few-shot divergono nella classificazione"""

    # identifica i sample con predizioni diverse tra i due approcci
    disagreements = [
        r for r in results
        if r["zero_shot"]["label"] != r["few_shot"]["label"]
    ]

    print(f"\ncasi in cui le due strategie divergono: {len(disagreements)}")

    for r in disagreements:
        zs_l = r["zero_shot"]["label"]
        fs_l = r["few_shot"]["label"]
        gt = r["ground_truth"]

        winner = (
            "few-shot"
            if fs_l == gt
            else ("zero-shot" if zs_l == gt else "nessuno")
        )

        print(
            f"  [id {r['id']}] "
            f"ZS={zs_l} | FS={fs_l} | GT={gt} → corretto: {winner}"
        )

        print(f"    {r['text_preview'][:120]}...")


# ─── entrypoint ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # conta le run disponibili per determinare la modalità di analisi
    available = count_runs()

    if available == 0:
        print("nessuna run disponibile in results/")
        sys.exit(1)

    # modalità di analisi multipla con aggregazione statistica
    if "--runs" in sys.argv:

        idx = sys.argv.index("--runs")

        if idx + 1 < len(sys.argv) and sys.argv[idx + 1].isdigit():
            n = min(int(sys.argv[idx + 1]), available)
        else:
            # utilizza tutte le run disponibili se non specificato
            n = available

        print(f"analisi di {n} run (disponibili: {available})")

        # carica i risultati e calcola le metriche aggregate
        runs = load_runs(n)

        zs = compute_multi_metrics(runs, "zero_shot")
        fs = compute_multi_metrics(runs, "few_shot")

        print_multi_metrics("zero-shot", zs)
        print_multi_metrics("few-shot", fs)

        # calcola il miglioramento del few-shot rispetto allo zero-shot
        delta = fs["mean_accuracy"] - zs["mean_accuracy"]

        direction = "↑" if delta > 0 else ("↓" if delta < 0 else "=")

        print(f"\n  delta medio (few-shot - zero-shot): {delta:+.2f} {direction}")

        print_stability(zs, fs)

        # dettaglio degli errori nella run finale
        print(f"\n\ndettaglio errori — run {n}:")

        print("\nerrori zero-shot:")
        print_errors(runs[-1], "zero_shot")

        print("\nerrori few-shot:")
        print_errors(runs[-1], "few_shot")

        print_disagreements(runs[-1])

    else:
        # modalità di analisi singola
        # analizza una singola run specificata o la prima disponibile

        run_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1

        if run_id > available:
            print(f"run {run_id} non disponibile (disponibili: {available})")
            sys.exit(1)

        print(f"analisi run {run_id}")

        results = load_run(run_id)

        # calcola le metriche per entrambe le strategie
        zs = compute_metrics(results, "zero_shot")
        fs = compute_metrics(results, "few_shot")

        print_metrics("zero-shot", zs)
        print_metrics("few-shot", fs)

        delta = fs["accuracy"] - zs["accuracy"]

        direction = "↑" if delta > 0 else ("↓" if delta < 0 else "=")

        print(f"\n  delta (few-shot - zero-shot): {delta:+.2f} {direction}")

        # analisi dettagliata degli errori
        print("\n\nerrori zero-shot:")
        print_errors(results, "zero_shot")

        print("\nerrori few-shot:")
        print_errors(results, "few_shot")

        print_disagreements(results)