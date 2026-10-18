// Command distill makes a smaller model from a trained one: the MLP of
// each layer of the encoder is pruned to its most useful neurons, measured
// on training texts, then the pruned model is trained on the labels and
// on the tags of the original, its teacher.
//
//	distill -teacher runs/work-fam -keep 576 -train data/synth2/train_fam.jsonl -out runs/work-fam-p576
package main

import (
	"context"
	"flag"
	"log"
	"math/rand"
	"time"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

func main() {
	teacherDir := flag.String("teacher", "runs/work-fam", "trained model to distill")
	keep := flag.Int("keep", 576, "MLP neurons kept in each layer")
	trainPath := flag.String("train", "data/synth2/train_fam.jsonl", "training examples")
	testPath := flag.String("test", "data/prose/fam-test.jsonl", "test examples, measured at the end")
	calib := flag.Int("calib", 300, "training texts drawn at random to measure the MLP neurons")
	epochs := flag.Int("epochs", 1, "epochs of distillation (0: prune only)")
	batch := flag.Int("batch", 8, "batch size")
	lr := flag.Float64("lr", 5e-5, "encoder learning rate")
	headLR := flag.Float64("head-lr", 1e-4, "head learning rate")
	alpha := flag.Float64("alpha", 0.5, "weight of the teacher's tags against the labels")
	temp := flag.Float64("temp", 2, "distillation temperature")
	hidden := flag.Float64("hidden", 1, "weight of the distance between the final states")
	seed := flag.Int64("seed", 1, "seed")
	from := flag.String("from", "", "start from this pruned model instead of pruning the teacher")
	out := flag.String("out", "", "model directory (required)")
	flag.Parse()
	if *out == "" {
		log.Fatal("-out is required")
	}

	train, err := dataset.ReadFile(*trainPath)
	if err != nil {
		log.Fatal(err)
	}
	test, err := dataset.ReadFile(*testPath)
	if err != nil {
		log.Fatal(err)
	}
	teacher, err := indecis.Load(*teacherDir, indecis.WithInt8())
	if err != nil {
		log.Fatal(err)
	}
	studentDir := *teacherDir
	if *from != "" {
		studentDir = *from
	}
	student, err := indecis.Load(studentDir)
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	rng := rand.New(rand.NewSource(*seed))
	var texts []string
	for _, i := range rng.Perm(len(train))[:min(*calib, len(train))] {
		texts = append(texts, train[i].Text)
	}
	if *from == "" {
		t0 := time.Now()
		if err := student.PruneMLP(ctx, texts, *keep); err != nil {
			log.Fatal(err)
		}
		log.Printf("pruned to %d MLP neurons per layer in %s", *keep, time.Since(t0).Round(time.Second))
	}

	if *epochs > 0 {
		opts := indecis.DefaultTrainOptions()
		opts.Epochs, opts.BatchSize, opts.LR, opts.HeadLR, opts.Seed = *epochs, *batch, *lr, *headLR, *seed
		opts.Teacher, opts.Distill, opts.DistillTemp, opts.DistillHidden = teacher, *alpha, *temp, *hidden
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
		log.Printf("distilling on %d examples, %d epochs", len(train), *epochs)
		if err := student.Fit(ctx, train, opts); err != nil {
			log.Fatal(err)
		}
	}
	// The teacher's temperatures rather than temperatures calibrated on
	// the labels: the model learned the teacher's logits, and a decoding
	// bias keeps the effect it has on the teacher, so the settings of the
	// model it replaces still hold.
	temps := teacher.Temperatures()
	if err := student.SetTemperatures(temps); err != nil {
		log.Fatal(err)
	}
	log.Printf("temperatures: %v", temps)
	if err := student.Save(*out); err != nil {
		log.Fatal(err)
	}
	metrics, err := student.Evaluate(ctx, test)
	if err != nil {
		log.Fatal(err)
	}
	for _, mt := range metrics {
		log.Printf("%s: %s", *testPath, mt)
	}
}
