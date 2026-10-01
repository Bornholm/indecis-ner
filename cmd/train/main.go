// Command train fine-tunes an indecis model that tags the entities of
// prepare's examples: one Spans question, "entities", with the WikiNER types.
//
//	train -backbone ~/.cache/indecis/models/bekko-embedding-v1-a8m -limit 40000 -out runs/a8m-40k
package main

import (
	"context"
	"flag"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

func main() {
	home, _ := os.UserHomeDir()
	backbone := flag.String("backbone", home+"/.cache/indecis/models/bekko-embedding-v1-a8m", "backbone directory")
	trainPath := flag.String("train", "data/train.jsonl", "training examples")
	devPath := flag.String("dev", "data/dev.jsonl", "calibration examples")
	testPath := flag.String("test", "data/test_clean.jsonl", "test examples, measured at the end")
	limit := flag.Int("limit", 0, "train on this many examples drawn at random (0: all)")
	maxLen := flag.Int("maxlen", 128, "tokens per window")
	epochs := flag.Int("epochs", 2, "epochs")
	batch := flag.Int("batch", 16, "batch size")
	lr := flag.Float64("lr", 1e-4, "encoder learning rate")
	headLR := flag.Float64("head-lr", 1e-3, "head learning rate")
	seed := flag.Int64("seed", 1, "seed")
	group := flag.Float64("group", 0, "share of the sentences joined into texts of 2 to 8 sentences, so that windows hold several")
	out := flag.String("out", "", "model directory (required)")
	flag.Parse()
	if *out == "" {
		log.Fatal("-out is required")
	}

	train, err := dataset.ReadFile(*trainPath)
	if err != nil {
		log.Fatal(err)
	}
	if *limit > 0 && *limit < len(train) {
		rng := rand.New(rand.NewSource(*seed))
		rng.Shuffle(len(train), func(i, j int) { train[i], train[j] = train[j], train[i] })
		train = train[:*limit]
	}
	if *group > 0 {
		train = groupSentences(train, *group, rand.New(rand.NewSource(*seed+1)))
	}
	dev, err := dataset.ReadFile(*devPath)
	if err != nil {
		log.Fatal(err)
	}
	test, err := dataset.ReadFile(*testPath)
	if err != nil {
		log.Fatal(err)
	}

	schema := indecis.Schema{indecis.NewSpans("entities", "Named entities: people, places, organizations, other proper names", "PER", "LOC", "ORG", "MISC")}
	m, err := indecis.New(*backbone, schema, *seed, indecis.WithMaxLen(*maxLen))
	if err != nil {
		log.Fatal(err)
	}
	opts := indecis.DefaultTrainOptions()
	opts.Epochs, opts.BatchSize, opts.LR, opts.HeadLR, opts.Seed = *epochs, *batch, *lr, *headLR, *seed
	var sum float64
	var n int
	opts.Progress = func(p indecis.Progress) {
		sum += p.Loss
		n++
		if p.Step%100 == 0 || p.Step == p.Steps {
			log.Printf("epoch %d step %d/%d loss %.4f lr %.2g %.0f tok/s eta %s", p.Epoch+1, p.Step, p.Steps, sum/float64(n), p.LR,
				float64(p.Tokens)/p.Elapsed.Seconds(), (time.Duration(float64(p.Elapsed) / float64(p.Step) * float64(p.Steps-p.Step))).Round(time.Second))
			sum, n = 0, 0
		}
	}
	ctx := context.Background()
	log.Printf("training on %d examples, %d epochs, windows of %d tokens", len(train), *epochs, *maxLen)
	if err := m.Fit(ctx, train, opts); err != nil {
		log.Fatal(err)
	}
	temps, err := m.Calibrate(ctx, dev)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("temperature: %.3f", temps["entities"])
	if err := m.Save(*out); err != nil {
		log.Fatal(err)
	}
	metrics, err := m.Evaluate(ctx, test)
	if err != nil {
		log.Fatal(err)
	}
	for _, mt := range metrics {
		log.Printf("%s: %s", *testPath, mt)
	}
}

// groupSentences joins a share of the sentences into texts of 2 to 8
// sentences, separated by a space, their entities shifted. Each sentence
// is used once: the training cost does not change.
func groupSentences(ex []dataset.Example, share float64, rng *rand.Rand) []dataset.Example {
	var out, pending []dataset.Example
	flush := func() {
		if len(pending) == 0 {
			return
		}
		var b strings.Builder
		var ents []any
		for i, e := range pending {
			if i > 0 {
				b.WriteByte(' ')
			}
			base := b.Len()
			b.WriteString(e.Text)
			list, _ := e.Labels["entities"].([]any)
			for _, item := range list {
				m := item.(map[string]any)
				ents = append(ents, map[string]any{"start": m["start"].(float64) + float64(base), "end": m["end"].(float64) + float64(base), "type": m["type"]})
			}
		}
		if ents == nil {
			ents = []any{}
		}
		out = append(out, dataset.Example{Text: b.String(), Labels: map[string]any{"entities": ents}})
		pending = nil
	}
	size := 0
	for _, e := range ex {
		if rng.Float64() >= share {
			out = append(out, e)
			continue
		}
		if size == 0 {
			size = 2 + rng.Intn(7)
		}
		pending = append(pending, e)
		if len(pending) == size {
			flush()
			size = 0
		}
	}
	flush()
	log.Printf("%d sentences grouped into %d texts", len(ex), len(out))
	return out
}
