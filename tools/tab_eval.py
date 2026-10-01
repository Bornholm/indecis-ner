"""Runs TAB's evaluation script (data/tab/evaluation.py) on one masked
output file: its command line passes a list where a path is expected.

    .venv/bin/python tools/tab_eval.py data/tab/echr_test.json runs/tab-a8m/masked_test.json
"""
import json, sys
sys.path.insert(0, "data/tab")
import evaluation as ev

gold, masked = sys.argv[1], sys.argv[2]
corpus = ev.GoldCorpus(gold)
docs = ev.get_masked_docs_from_file(masked)
m = ev.evaluate(corpus, docs)
keys = ["token_recall", "recall_direct_entities", "recall_quasi_entities", "mention_recall", "token_precision", "mention_precision", "token_f1"]
print(json.dumps({k: m[k] for k in keys} | {"token_recall_by_type": m["token_recall_by_type"]}))
