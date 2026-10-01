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
