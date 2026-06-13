"""Train (or retrain) the LightGBM LambdaRank ranker on the full pool."""
import sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from redrob import config
from redrob.data import load_or_build_parquet
from redrob.rank.train import train_ranker, build_feature_frame, FEATURE_COLUMNS

if __name__ == "__main__":
    t = time.time()
    df = load_or_build_parquet()
    print(f"parquet: {len(df):,}")
    df = build_feature_frame(df)
    print(f"features built in {time.time()-t:.1f}s")
    t = time.time()
    model, feats = train_ranker(force=True)
    print(f"trained in {time.time()-t:.1f}s; {len(feats)} features; saved -> {config.RANKER_TXT}")
