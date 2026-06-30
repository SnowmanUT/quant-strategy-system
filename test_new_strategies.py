"""
Targeted backtest for the 6 newly added strategy families.
Uses the same walk-forward + six-filter funnel as the main sweep.
"""
import warnings
warnings.filterwarnings("ignore")

from layer1 import download_data, build_configs
from layer2 import run_pipeline, funnel_report

NEW_FAMILIES = {
    "cmo_fisher",
    "rvi_divergence",
    "adaptive_rsi",
    "elder_impulse",
    "klinger_osc",
    "price_oscillator",
}

if __name__ == "__main__":
    universe = download_data()
    all_configs = build_configs()

    new_configs = [c for c in all_configs if c[1].__name__ in NEW_FAMILIES]
    print(f"\n{len(new_configs)} configs × {len(universe)} assets "
          f"= {len(new_configs)*len(universe):,} backtests\n")

    results = run_pipeline(
        universe,
        new_configs,
        cache=False,
        csv_path="new_strategies_batch7.csv",
        pkl_path="data_cache/new_strategies_batch7.pkl",
    )

    survivors = funnel_report(results, top_n=20)
