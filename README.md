# indecis-ner

Can an [indecis](../indecis) model replace the CRF of [go-anon](../go-anon)? This project trains a `spans` model on go-anon's French WikiNER corpus and runs both engines on the same raw texts.

```bash
make data      # WikiNER from go-anon, deduplicated
make train     # about 1.5 hours on a laptop, all cores
make compare compare-docs
```

## The test set

85% of the sentences of go-anon's French test set (2,256 of 2,664) also appear in its training set, and the CRF was trained on train + dev. On the whole test set it scores F1 0.94; on the 367 test sentences it has never seen, 0.80. `prepare` therefore writes:

- `train.jsonl`: the 143,749 distinct training sentences, without any test or dev sentence;
- `test_clean.jsonl`: the 367 test sentences absent from train and dev (821 entities), the only fair test for the CRF;
- `test_full.jsonl`: the whole test set, comparable to the published scores;
- `docs.jsonl`: the clean sentences joined into 7 documents of about 1,500 words.

An entity counts when its type and both bounds are exact. The lenient recall only asks for the type and an overlap, which is what masking needs.

## Results

October 1, 2026, Core Ultra 7 265U, one core per text. The CRF runs with the gazetteers and Brown clusters it was trained with. indecis: bekko-embedding-v1-a8m, 2 epochs, windows of 128 tokens, half of the sentences joined into texts of 2 to 8 sentences (`-group 0.5`), int8.

| Clean test | CRF | indecis |
| --- | --- | --- |
| F1, all | 0.797 | 0.810 |
| F2, all | 0.791 | 0.811 |
| PER | 0.919 | 0.921 |
| LOC | 0.796 | 0.790 |
| ORG | 0.719 | 0.725 |
| MISC | 0.639 | 0.734 |
| Lenient recall | 0.895 | 0.920 |
| Time per sentence | 8.0 ms | 4.4 ms |

| Documents of 1,500 words | CRF | indecis |
| --- | --- | --- |
| F1, all | 0.797 | 0.802 |
| PER | 0.919 | 0.932 |
| Lenient recall | 0.895 | 0.909 |
| Time per kB | 50 ms | 30 ms |

With 821 entities, a difference of one or two points is within noise.

- Trained on single sentences only, the model lost 4.5 points on documents (0.772): it had never seen a window holding several sentences. Joining sentences at training time removed the loss, at the same cost.
- In float32, quality is the same as int8 and an indecis sentence takes 9.9 ms.
- Token boundaries cost nothing: tagging the labeled entities on BPE tokens and decoding them back rebuilds 100% of the test entities exactly (`make ceiling`).

## Indirect identification: TAB

The [Text Anonymization Benchmark](https://github.com/NorskRegnesentral/text-anonymization-benchmark) (Pilán et al. 2022, MIT) annotates 1,268 rulings of the European Court of Human Rights, in English, about 1,200 tokens each. Every mention has a type (PERSON, CODE, LOC, ORG, DEM, DATETIME, QUANTITY, MISC) and a masking decision for the person to protect: DIRECT identifier, QUASI-identifier, or nothing to mask. Quasi-identifiers are what go-anon does not cover: dates, professions, places and relatives that together single someone out.

The model has two Spans questions on one encoder: `entities` for the types, `mask` for DIRECT and QUASI. bekko a8m, windows of 512 tokens, 3 epochs on the 1,014 training rulings (about one hour), int8. Scores come from TAB's own evaluation script (`make tab-eval`): token-level recall, entity-level recall of direct and quasi-identifiers, uniform token precision.

`WithSpanBias` adds a bias towards passages at decoding, the counterpart of the loss weights the Longformer was trained with. The bias, 2, was chosen on the dev set, where it gives the Longformer's recall (0.934 against 0.935).

| Test set (127 rulings) | Recall | Direct | Quasi | Precision |
| --- | --- | --- | --- | --- |
| indecis, no bias | 0.843 | 0.984 | 0.809 | 0.909 |
| indecis, bias 2 | 0.934 | 0.992 | 0.926 | 0.768 |
| Longformer (149M parameters, 4,096 tokens, weights 10:1), from the paper | 0.919 | 1.000 | 0.916 | 0.836 |
| Presidio, from the paper | 0.707 | 0.460 | 0.758 | 0.761 |

DEM and MISC are the hardest types, as in the paper: token recall 0.67 and 0.52 with the bias.

### In French

There is no French counterpart of TAB. `make tab-translate` has Claude translate the test rulings into French, each mention of the first annotator wrapped in numbered markers that the translation keeps: the French texts carry the same decisions without being annotated again. 126 of the 127 rulings came back with all their markers. The English reference is the same 126 rulings, first annotator only.

| First annotator, 126 rulings | Recall | Direct | Quasi | Precision |
| --- | --- | --- | --- | --- |
| English, bias 2 | 0.948 | 0.992 | 0.941 | 0.765 |
| French, bias 2 | 0.875 | 0.977 | 0.884 | 0.709 |
| French, no bias | 0.701 | 0.850 | 0.703 | 0.923 |

The model never saw French annotations: the multilingual backbone carries most of the task over, and loses 6 points of quasi-identifier recall. ORG (0.85 to 0.62) and DEM (0.70 to 0.46) lose the most.

## Using it in go-anon

go-anon's anonymizer takes any `anonymizer.Recognizer`. `recognizer.New` wraps an indecis model and applies go-anon's filters after it, as its own recognizer does after the CRF:

```go
m, _ := indecis.Load("runs/a8m-group", indecis.WithInt8())
r := recognizer.New(m, "entities",
	ner.RegexEntityFilter(ner.BuiltinRegexPatterns),
	ner.RegexEntityFilter(ner.SecretPatterns()))
anon := anonymizer.New(r, anonymizer.Config{Strategy: anonymizer.TagReplace, ConsistentMap: true})
```

`TestAnonymizeWithIndecis` runs it with strict verification and checks the round trip.

## Status

Depends on the `feat/spans` branch of indecis (`replace` directives in `go.mod`). The model file is 305 MB (bf16 embedding table), go-anon's French CRF 278 MB.
