GO := GOEXPERIMENT=simd go
GOANON := ../go-anon
GAZ := $(HOME)/.cache/go-anon/models
CRF_FLAGS := -crf $(GOANON)/models/model_fr_pruned.crf.gz \
	-gazetteers firstnames:$(GAZ)/firstnames.txt,locations:$(GAZ)/locations.txt \
	-clusters $(GOANON)/data/fr/brown_clusters_fr.txt
MODEL ?= runs/a8m-full

.PHONY: build data train compare compare-docs ceiling test

build:
	$(GO) build -o bin/ ./cmd/...

# WikiNER from go-anon, deduplicated: see cmd/prepare.
data: build
	./bin/prepare -data $(GOANON)/data/fr -out data

train: build
	./bin/train -epochs 2 -out $(MODEL)

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
