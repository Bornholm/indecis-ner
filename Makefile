GO := GOEXPERIMENT=simd go
GOANON := ../go-anon
GAZ := $(HOME)/.cache/go-anon/models
CRF_FLAGS := -crf $(GOANON)/models/model_fr_pruned.crf.gz \
	-gazetteers firstnames:$(GAZ)/firstnames.txt,locations:$(GAZ)/locations.txt \
	-clusters $(GOANON)/data/fr/brown_clusters_fr.txt
MODEL ?= runs/a8m-group

.PHONY: build data train compare compare-docs ceiling test

build:
	$(GO) build -o bin/ ./cmd/...

# WikiNER from go-anon, deduplicated: see cmd/prepare.
data: build
	./bin/prepare -data $(GOANON)/data/fr -out data

train: build
	./bin/train -epochs 2 -group 0.5 -out $(MODEL)

# Both engines on the clean test, then on the whole test (contaminated
# for the CRF, comparable to go-anon's published scores).
compare: build
	./bin/compare -test data/test_clean.jsonl $(CRF_FLAGS) -indecis $(MODEL) -json $(MODEL)/compare_clean.json
	./bin/compare -test data/test_clean.jsonl -indecis $(MODEL) -int8
	./bin/compare -test data/test_full.jsonl $(CRF_FLAGS) -indecis $(MODEL) -json $(MODEL)/compare_full.json

compare-docs: build
	./bin/compare -test data/docs.jsonl $(CRF_FLAGS) -indecis $(MODEL) -json $(MODEL)/compare_docs.json

ceiling: build
	./bin/ceiling -test data/test_clean.jsonl
	./bin/ceiling -test data/train.jsonl -show 0

test:
	$(GO) test ./...

# TAB (data/tab, see cmd/tab): masking decisions scored by TAB's own
# evaluation script, on the English test set, its first annotator only,
# and its French translation (cmd/translate).
TAB_MODEL ?= runs/tab-a8m
PY := .venv/bin/python

tab-data: build
	for s in train dev test; do ./bin/tab convert -in data/tab/echr_$$s.json -out data/tab/$$s.jsonl; done

tab-train: build
	./bin/train -task tab -train data/tab/train.jsonl -dev data/tab/dev.jsonl -test data/tab/test.jsonl -maxlen 512 -batch 8 -epochs 3 -out $(TAB_MODEL)

TAB_BIAS ?= 2

tab-eval: build
	for g in test test_a1 test_fr; do \
		for b in 0 $(TAB_BIAS); do \
			./bin/tab predict -in data/tab/echr_$$g.json -model $(TAB_MODEL) -int8 -bias $$b -out $(TAB_MODEL)/masked_$${g}_b$$b.json && \
			echo "$$g bias $$b: $$($(PY) tools/tab_eval.py data/tab/echr_$$g.json $(TAB_MODEL)/masked_$${g}_b$$b.json | tail -1)"; \
		done; \
	done

# French translation of the test set (Claude, about 130 calls).
tab-translate: build
	./bin/translate -in data/tab/echr_test.json -out data/tab/echr_test_fr.json -orig data/tab/echr_test_a1.json
