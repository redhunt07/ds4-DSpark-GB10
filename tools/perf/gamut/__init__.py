"""gamut — ds4 GB10 performance suite (capture · analyze · store · compare).

A single Python package replacing the old tools/perf bash + script sprawl:

  gamut capture   nsys-trace a ds4 run, join ptxas/gb20b/ncu/MTP telemetry,
                  build a report, ingest to the run-store          (was capture.sh)
  gamut bench     throughput matrix (plain/mtp cells x iters) under a
                  GPU/thermal monitor                              (was bench-with-monitor.sh)
  gamut report    analyze an existing nsys sqlite -> md/json/html  (was gamut.py)
  gamut db        run-store: list / show / compare / backfill      (was gamut_db.py)

Pure stdlib (sqlite3, subprocess, threading). GB10 constants are calibrated
by tools/perf/membw.cu, not guessed.
"""

__version__ = "1.0.0"
