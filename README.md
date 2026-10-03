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

Adding 195 training rulings translated the same way (a random sample of 200, 5 lost markers three times and were left out; `runs/chain_fr200.sh`) to the 1,014 English ones:

| First annotator, 126 rulings, bias 2 | Recall | Direct | Quasi | Precision | ORG | DEM |
| --- | --- | --- | --- | --- | --- | --- |
| English, English training | 0.948 | 0.992 | 0.941 | 0.765 | 0.849 | 0.698 |
| English, + 195 French | 0.951 | 0.996 | 0.944 | 0.758 | 0.851 | 0.735 |
| French, English training | 0.875 | 0.977 | 0.884 | 0.709 | 0.615 | 0.461 |
| French, + 195 French | 0.913 | 0.985 | 0.917 | 0.720 | 0.792 | 0.519 |

ORG and DEM are token recalls. A fifth of the training set in French halves the gap with English (quasi-identifiers: 5.7 points down to 2.7) and costs English nothing. On the whole English test set (all annotators), the model scores 0.938 recall, 0.993 direct, 0.931 quasi, 0.762 precision.

## Workplace documents

`GUIDELINES.md` adapts TAB's rules to French workplace documents: every natural person is protected, and a third Spans question, `sensitive`, flags the special categories of articles 9 and 10 GDPR (health, union, beliefs, politics, origin, sexuality, offences). `synth` writes documents labeled by construction from templates (`synth/templates`), reusing go-anon's value generators for identities, organisations, addresses and identifiers: payslip, employment contract, prescription, sick note, absence email, disciplinary letter, appraisal, staff representative's email, and, kept for testing only, fitness certificate and work certificate.

`runs/work-a8m` trains the three questions on TAB, its 195 French rulings and 2,000 synthetic documents (`-task work`, about 1h45). Masking, int8, any decision (`compare -question mask -merge`):

| Synthetic test, 200 documents | Exact recall | Precision | Lenient recall |
| --- | --- | --- | --- |
| Templates seen in training, TAB model | 0.569 | 0.344 | 0.918 |
| Templates seen in training, work model | 0.999 | 0.998 | 1.000 |
| Templates never seen, TAB model | 0.527 | 0.310 | 0.880 |
| Templates never seen, work model | 0.850 | 0.514 | 0.985 |

- On seen templates the model has learned the templates: that score measures memory, not generalization.
- On the two unseen templates, nearly every passage to mask is touched (0.985). Most exact misses are conventions, not leaks: "Mme Martin Chloé (Née Marchand)" masked as two passages, a company name masked without its legal form, a home address marked QUASI where the guidelines say DIRECT. The lost precision is partly masks the guidelines do not ask for (the employer's address and SIRET, "Médecin du travail"), partly noise (a lone "(", "Apte").
- Special categories do not generalize: on unseen templates, HEALTH recall is 0.27 for 0.08 precision. The template lists are too small for the model to learn what a health datum is rather than which ones exist.
- TAB is not hurt: at a bias of 3 (the synthetic dev set changed the calibration), English test recall 0.931, quasi 0.922, precision 0.785; French quasi 0.909, precision 0.738, as without the synthetic documents.

### Documents in prose

Templates give structure, not language. `prose` has Claude Sonnet write documents in prose (emails, appraisal and meeting reports, letters, notes, chat threads), its labels inline with markers ⟦TYPE/MASK/SENSITIVE⟧…⟦/⟧ following `GUIDELINES.md`, from a scenario (`synth/scenarios.tsv`) and a cast drawn by go-anon's generators so that names do not repeat. Pi (minimax) then checks the labels. The 10 test scenarios are genres absent from the 40 training ones: accident inquiry, harassment report, settlement agreement, testimony for an employment tribunal, social aid committee...

388 of 400 training documents and 98 of 100 test documents came back with well-formed markers (a second pass retried the malformed ones). Pi changed no masking decision on 240 and 56 of them; its remarks are in `data/prose/*_review.jsonl`.

`runs/work-prose` adds the 388 training documents to `runs/work-a8m`'s data. Masking, any decision, int8, on the 98 test documents. Lenient: a passage counts as found if a mask overlaps it, and a mask counts as right if it overlaps a labeled passage.

| Prose test, 98 documents, bias 2 | Lenient recall | Lenient precision | Exact recall | Exact precision |
| --- | --- | --- | --- | --- |
| work-a8m (TAB, templates) | 0.875 | 0.696 | 0.657 | 0.481 |
| work-prose (+ prose) | 0.957 | 0.782 | 0.745 | 0.574 |

On the 56 documents Pi agreed with, the scores are the same within a point.

| Special categories, prose test | Labeled | Lenient recall | Lenient precision |
| --- | --- | --- | --- |
| HEALTH, work-a8m | 132 | 0.765 | 0.349 |
| HEALTH, work-prose | 132 | 0.712 | 0.717 |
| UNION, work-prose | 17 | 0.706 | 0.923 |
| BELIEF, work-prose | 23 | 0 | - |

- Prose training documents lift recall on unseen genres from 0.875 to 0.957, with fewer extra masks.
- Health data are found about as often but far more precisely. Beliefs are never found: the training data hold too few of them (77 template passages).
- Exact bounds stay hard in prose: is the passage "asthme" or "trois crises d'asthme sévères en un mois"? Masking needs the overlap, which is why lenient scores are the ones to read here.
- Nothing is lost elsewhere: unseen templates 0.977 lenient recall (0.994 before), TAB at bias 3 English quasi 0.929 and French 0.920, precision 0.761 and 0.709.

### Rarer special categories

Beliefs, politics, origin, sexuality and offences were missing or nearly. They now have value lists (`synth/lists`), three training templates (absence request, discrimination report, candidate summary), sections in two others, and a test-only template (confidential HR note); and 20 prose scenarios aimed at them, 4 of them for testing (`cat-train`, `cat-test` in `synth/scenarios.tsv`): 77 training and 20 test documents. A generator bug fixed on the way glued units to numbers ("3mois"); the synthetic corpus was regenerated (`data/synth2`).

`runs/work-cat` trains on everything. Special categories, lenient recall / lenient precision:

| | Targeted prose test (20 docs) | General prose test (98 docs) | Unseen templates (300 docs) |
| --- | --- | --- | --- |
| BELIEF | 0.18 / 0.40 (11) | 0.83 / 0.86 (23) | 0.90 / 0.98 (86) |
| POLITICS | 0.71 / 0.71 (7) | - | 1.00 / 1.00 (42) |
| ORIGIN | 0.25 / 1.00 (12) | - | 1.00 / 1.00 (29) |
| SEXUALITY | - | - | 1.00 / 1.00 (21) |
| OFFENCE | 0.80 / 0.90 (10) | - | 1.00 / 0.96 (51) |
| HEALTH | - | 0.76 / 0.70 (132) | 0.40 / 0.45 (53) |
| UNION | 1.00 / 1.00 (5) | 0.71 / 0.93 (17) | - |

Labeled passages in parentheses. `runs/work-prose` found none of BELIEF, POLITICS, ORIGIN, SEXUALITY and OFFENCE on any of these sets.

- The unseen templates draw their values from the same lists as the training templates: perfect scores there say the model recognizes the listed phrasings, not that it generalizes. The targeted prose test is the honest one, and it is small: 7 to 12 passages per category.
- Offences and politics carry over to new prose; beliefs and origin barely do (0.18 and 0.25 recall). Beliefs did carry over to the general prose test (0.83), where they are stated the way the lists state them.
- Masking and TAB are unchanged: general prose test 0.962 lenient recall for 0.762 precision, unseen templates 0.988, TAB at bias 3 English quasi 0.929, French 0.920.

### Words never seen in training

A test that reuses the training vocabulary measures word recognition. `lexicon` records, from all the training data, the words that signal each special category: a stem (first six letters, without accents) that occurs at least twice in the category's passages, with at least 30% of its occurrences in the training text inside them ("mosquée" signals beliefs, "chaque" or "midi" do not). `compare -novel` then scores only the test passages that contain none of these words, and ignores the others along with what the model finds on them.

Measured that way, `runs/work-cat` found **none** of the novel passages of the targeted prose test: its scores came from the words.

So 150 more training documents were written with indirect phrasings (fasting, a pilgrimage, a prison term, leafleting, a husband in a wedding photo: scenarios `ind-train`, 144 kept), the lexicon was rebuilt over all the training data, and 60 test documents (`ind-test`, 56 kept) were written with its 100 most frequent signalling words forbidden. Of their 244 sensitive passages, 114 contain no signalling word. `runs/work-ind` adds the indirect training documents. Lenient recall / precision:

| `ind-test` | work-cat, all | work-ind, all | work-cat, novel | work-ind, novel |
| --- | --- | --- | --- | --- |
| BELIEF | 0.08 / 1.00 (49) | 0.33 / 1.00 | 0 (35) | 0.26 / 1.00 |
| OFFENCE | 0.07 / 1.00 (28) | 0.39 / 0.83 | 0 (13) | 0.08 / 0.25 |
| ORIGIN | 0 (13) | 0.08 / 1.00 | 0 (12) | 0.08 / 1.00 |
| POLITICS | 0.13 / 1.00 (15) | 0.13 / 0.67 | 0 (13) | 0 |
| SEXUALITY | 0 (14) | 0.36 / 0.55 | 0 (10) | 0.20 / 0.29 |
| HEALTH | 0.41 / 0.89 (69) | 0.86 / 0.83 | 0 (10) | 0.20 / 0.14 |
| UNION | 0.46 / 0.97 (56) | 0.91 / 0.95 | 0.24 / 0.83 (21) | 0.95 / 0.88 |

- Indirect training documents help, but classifying a special category from words never seen with it stays out of reach: 0 to 26% recall on novel passages, union membership excepted.
- The passages are masked all the same. A mask overlaps 91% of the sensitive passages of `ind-test` and 78% of the novel ones (work-ind, bias 2): the model sees personal data there, and fails to say which special category it falls under.
- Masking and TAB are unchanged: `ind-test` 0.956 lenient recall, general prose test 0.959, TAB at bias 3 English quasi 0.933, French 0.922.

### Leaks

What matters is how much personal data stays readable, not whether each passage gets the right type or exact bounds. `compare -question mask,sensitive -merge` scores the union of the two questions against the union of their labels, and reports characters: the share of the characters of labeled passages left unmasked (the leak), the share of masked characters that belong to a labeled passage, and the documents with no leak. Spaces, punctuation and function words (articles, prepositions, possessives) are not counted, as TAB leaves them out: an unmasked "le" before a masked "service de pneumologie" leaks nothing.

`-refine` applies what the go-anon adapter does (`recognizer.Masks`): masks grow to whole words ("[[é]]crasé" is masked as "écrasé"), masks of function words alone are dropped, and masks separated only by function words and commas are joined ("magasinier sur le site de Stains" becomes one mask), except across a sentence and except for names and identifiers, which keep their own placeholder.

`runs/work-ind`, int8, refined, by decoding bias:

| Test set | Bias 0 | Bias 4 | Bias 5 | Bias 6 |
| --- | --- | --- | --- | --- |
| Prose, 98 docs: leaked / masked right / docs without a leak | 15.8% / 86% / 2 | 3.0% / 68% / 49 | 2.0% / 63% / 61 | 1.3% / 58% / 73 |
| Indirect prose, 56 docs | 19.0% / 82% / 2 | 6.2% / 66% / 11 | 3.9% / 61% / 18 | 2.4% / 57% / 27 |
| Targeted prose, 20 docs | 14.2% / 80% / 1 | 5.8% / 62% / 4 | 3.4% / 57% / 9 | 2.1% / 52% / 14 |
| Unseen templates, 300 docs | 3.3% / 78% / 149 | 1.6% / 72% / 196 | 1.1% / 69% / 226 | 0.5% / 68% / 264 |

- The bias is the lever: from 0 to 6, the leak in prose falls from 16% to 1.3%, and the documents without any leak from 2 to 73 of 98.
- The cost is extra masks: at bias 5, about 40% of the masked characters belong to no labeled passage. Some are personal data the labels missed ("hospitalisé et son pronostic", the date of an accident); some are not.
- What still leaks at bias 6 (`compare -leaks`): relatives and family situation ("conjoint", "Parent isolé", "Sa femme"), some health details ("infiltration"), generic roles ("la trésorière du CSE"), and events ("une procédure d'expulsion").
- Bias 5 is a reasonable default for the go-anon adapter; 6 when every leak counts more than readability.

### Relatives, family situation, life events

What leaked most at bias 6 was relatives, family situations and life events. 26 scenarios aimed at them (bereavement, divorce and custody, eviction, debt, domestic violence, caring for a relative, adoption: `fam-train`, `fam-test`) gave 113 training and 29 test documents, the writer asked for at least five such passages each. `runs/work-fam` adds them to `runs/work-ind`'s data.

Characters leaked overstate what leaks in long passages: in "divorce [prononcé le] 12/02/2024", the unmasked words reveal nothing once the date and "divorce" are masked. `compare` therefore also counts the exposed passages, those of which less than half the counted characters are masked.

| Bias 6, refined | work-ind: chars leaked / passages exposed | work-fam: chars leaked / passages exposed |
| --- | --- | --- |
| Family test, 29 docs (720 passages) | 5.5% / 3.6% | 4.7% / 2.4% |
| Prose, 98 docs (2,054) | 1.3% / 1.1% | 1.1% / 0.9% |
| Indirect prose, 56 docs (1,053) | 2.4% / 1.1% | 2.6% / 1.3% |
| Unseen templates, 300 docs (3,033) | 0.5% / 0.2% | 0.7% / 0.4% |

- On family data, a third fewer passages are exposed; elsewhere the differences are a few passages either way. The share of masked characters that are labeled does not move (62% on the family test, 59% in prose at bias 6). TAB is unchanged (bias 3: English quasi 0.933, French 0.923).
- What is still exposed: situations stated in a few common words ("sans logement", "en arrêt prolongé", "La pension alimentaire", "a repris un emploi salarié"), and unnamed services ("Direction des ressources humaines", "la plateforme Indemnisation").
- `runs/work-fam` is now the default model of `indecis-anon`.

### A larger backbone: a25m

`runs/work-ind-a25m` is `runs/work-ind` trained on bekko-embedding-v1-a25m (13 layers instead of 4, same width): same data, same settings, about 6 hours instead of 2 (690 tokens per second, 6.6 GB at its peak).

At the same bias it masks less, so the comparison is made at equal leak (int8, refined, `-question mask,sensitive`):

| Test set | a8m | a25m |
| --- | --- | --- |
| Prose, 98 docs | bias 6: 1.3% leaked, 58% of the masked characters labeled | bias 7: 1.2% leaked, 66% labeled |
| Indirect prose, 56 docs | bias 6: 2.4%, 57% | bias 8: 2.3%, 60% |
| Targeted prose, 20 docs | bias 6: 2.1%, 52% | bias 8: 2.0%, 54% |
| Unseen templates, 300 docs | bias 6: 0.5%, 68% | bias 8: 1.3%, 69% |

| 10 A4 pages (30,000 characters), int8 | a8m | a25m |
| --- | --- | --- |
| One core | 1.3 s | 4.2 s |
| All cores | 0.66 s | 2.1 s |
| Memory after the document (resident / own) | 320 MB / 263 MB | 429 MB / 336 MB |

- For the same leak in prose, a25m masks 2 to 8 points fewer characters by mistake: a better reading, not fewer leaks.
- On the templates, what a25m leaves is mostly the legal form of companies ("au capital de"), which the labels include in the employer's passage: no personal data.
- Special categories on novel words barely move: belief 0.29, offence 0.39, origin 0.17, politics 0.08, sexuality 0.10 lenient recall.
- TAB at bias 3: English quasi 0.924 for precision 0.785, French 0.913 for 0.776 (a8m: 0.933 / 0.762 and 0.922 / 0.699).
- Three times slower for a modest gain: a8m at bias 5 or 6 stays the default; a25m is worth it where extra masks hurt the reading of the document.

## Using it in go-anon

go-anon's anonymizer takes any `anonymizer.Recognizer`. `recognizer.New` wraps an indecis model and applies go-anon's filters after it, as its own recognizer does after the CRF:

```go
m, _ := indecis.Load("runs/a8m-group", indecis.WithInt8())
r := recognizer.New(m, []string{"entities"},
	ner.RegexEntityFilter(ner.BuiltinRegexPatterns),
	ner.RegexEntityFilter(ner.SecretPatterns()))
r.Raw = true
anon := anonymizer.New(r, anonymizer.Config{Strategy: anonymizer.TagReplace, ConsistentMap: true})
```

`TestAnonymizeWithIndecis` runs it with strict verification and checks the round trip.

To mask personal and sensitive data rather than named entities, load a masking model with a bias and give the adapter the two mask questions; it joins and refines their passages and types each mask from the `entities` question (`TestMaskWithIndecis`):

```go
m, _ := indecis.Load("runs/work-ind", indecis.WithInt8(),
	indecis.WithSpanBias("mask", 5), indecis.WithSpanBias("sensitive", 5))
r := recognizer.New(m, []string{"mask", "sensitive"}, ner.RegexEntityFilter(ner.BuiltinRegexPatterns))
```

## Trying it on real documents

`indecis-anon` is go-anon's `anon-doc` with an indecis masking model in place of the CRF: the same formats (DOCX, ODT, PDF with OCR, CSV/TSV, plain text), the same verification, fail-closed mode and cleaning of hidden surfaces, go-anon's regular expressions for identifiers and secrets. go-anon itself is not modified.

```bash
make build
bin/indecis-anon -model runs/work-fam -bias 5 -strict -input note.docx -output note_anon.docx
```

`-bias` sets how far it masks (see Leaks): 5 by default, 6 when leaks matter more than readability. Not carried over from anon-doc: the hash strategy and the encrypted mapping store, internal to go-anon's commands (`-save-mapping-insecure` writes the mapping in clear).

## Status

Depends on the `feat/spans` branch of indecis (`replace` directives in `go.mod`). The model file is 305 MB (bf16 embedding table), go-anon's French CRF 278 MB.
