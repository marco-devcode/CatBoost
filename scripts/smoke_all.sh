#!/data/data/com.termux/files/usr/bin/bash
# Smoke test di TUTTI gli script di misura: nessuno deve fallire.
# Classe di bug da prevenire: uno script che stampa i risultati principali
# e poi muore su un'eccezione (TypeError, KeyError) nella diagnostica.
set -e
export PROOT_NO_SECCOMP=1
proot-distro login ubuntu -- /bin/bash -lc '
  cd /data/data/com.termux/files/home/_scan/CatBoost
  export PYTHONUNBUFFERED=1
  fail=0
  for s in src/implied.py src/advanced_features.py src/evaluate_market.py \
           src/diagnose_vs_market.py src/experiments.py src/eval_calibration.py \
           src/eval_missing_market.py src/market_blend.py \
           scripts/select_devig.py scripts/verify_edge.py scripts/test_implied.py; do
    if [ "$s" = "src/evaluate_market.py" ] || [ "$s" = "src/diagnose_vs_market.py" ] \
       || [ "$s" = "src/experiments.py" ] || [ "$s" = "src/eval_calibration.py" ] \
       || [ "$s" = "src/eval_missing_market.py" ] || [ "$s" = "src/market_blend.py" ] \
       || [ "$s" = "scripts/verify_edge.py" ]; then
      out=$(/opt/venv/bin/python $s --league seriea 2>&1); rc=$?
    else
      out=$(/opt/venv/bin/python $s 2>&1); rc=$?
    fi
    if [ $rc -ne 0 ]; then
      echo "FAIL (rc=$rc) $s"
      echo "$out" | tail -4
      fail=1
    else
      echo "OK   $s"
    fi
  done
  echo "=== smoke completato, fail=$fail ==="
  exit $fail
'
