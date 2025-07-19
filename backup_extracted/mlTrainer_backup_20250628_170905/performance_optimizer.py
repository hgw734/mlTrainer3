
import os
import gc
import tracemalloc
import psutil
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


def print_header(title):
    print("\n" + "=" * 60)
    print(f"🛠️  {title}")
    print("=" * 60)


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        "rss_mb": round(mem_info.rss / 1024**2, 2),
        "vms_mb": round(mem_info.vms / 1024**2, 2),
        "shared_mb": round(mem_info.shared / 1024**2, 2),
    }


def scan_module_sizes(limit=10):
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('filename')
    print_header("Top Memory Usage by File")
    for stat in top_stats[:limit]:
        print(
            f"{stat.traceback[0].filename:<50} | {round(stat.size / 1024, 1)} KB")


def optimize_system_settings():
    print_header("Optimizing System Libraries")
    try:
        import numpy as np
        os.environ["NUMEXPR_MAX_THREADS"] = "2"
        np.set_printoptions(precision=3, suppress=True)
        print("✅ NumPy optimized")
    except BaseException:
        print("⚠️ NumPy not available")

    try:
        import pandas as pd
        pd.set_option("mode.chained_assignment", None)
        pd.set_option("display.max_columns", 10)
        print("✅ pandas optimized")
    except BaseException:
        print("⚠️ pandas not available")

    try:
        import tensorflow as tf
        tf.config.optimizer.set_jit(True)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        print("✅ TensorFlow JIT & threading optimized")
    except BaseException:
        print("⚠️ TensorFlow not available")


def run_garbage_collector():
    print_header("Running Garbage Collection")
    collected = gc.collect()
    print(f"✅ Garbage collector reclaimed {collected} unreachable objects")


def memory_report():
    print_header("Final Memory Usage")
    usage = get_memory_usage()
    for k, v in usage.items():
        print(f"{k:<10}: {v} MB")


def main():
    print(f"🚀 Performance Optimizer Start - {datetime.utcnow().isoformat()}")
    tracemalloc.start()
    optimize_system_settings()
    run_garbage_collector()
    scan_module_sizes()
    memory_report()


if __name__ == "__main__":
    main()
