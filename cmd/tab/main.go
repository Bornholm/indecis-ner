// Command tab converts the Text Anonymization Benchmark (TAB, Pilán et al.
// 2022, MIT) into indecis examples, and writes a model's masking decisions
// in the format of its evaluation script.
//
//	tab convert -in data/tab/echr_train.json -out data/tab/train.jsonl
//	tab predict -in data/tab/echr_test.json -model runs/tab -out runs/tab/masked_test.json
//
// Each mention becomes two passages: its type in the "entities" question
// (PERSON, CODE, LOC, ORG, DEM, DATETIME, QUANTITY, MISC), and, if it must
// be masked, DIRECT or QUASI in the "mask" question. TAB counts offsets in
// characters, indecis in UTF-8 bytes: both directions are converted.
//
// A document annotated by several people is trained on its first
// annotator; the evaluation script reads them all.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"unicode/utf8"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

type mention struct {
	Type       string `json:"entity_type"`
	Start      int    `json:"start_offset"`
	End        int    `json:"end_offset"`
	Text       string `json:"span_text"`
	Identifier string `json:"identifier_type"`
}

type document struct {
	ID          string `json:"doc_id"`
	Text        string `json:"text"`
	Annotations map[string]struct {
		Mentions []mention `json:"entity_mentions"`
	} `json:"annotations"`
	Split string `json:"dataset_type"`
}

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: tab convert|predict [flags]")
	}
	fs := flag.NewFlagSet(os.Args[1], flag.ExitOnError)
	in := fs.String("in", "", "TAB JSON file")
	out := fs.String("out", "", "output file")
	model := fs.String("model", "", "indecis model (predict)")
	int8 := fs.Bool("int8", false, "int8 inference (predict)")
	question := fs.String("question", "mask", "spans question whose passages are masked (predict)")
	bias := fs.Float64("bias", 0, "decoding bias towards passages (predict, see indecis.WithSpanBias)")
	fs.Parse(os.Args[2:])
	docs, err := read(*in)
	if err != nil {
		log.Fatal(err)
	}
	switch os.Args[1] {
	case "convert":
		err = convert(docs, *out)
	case "predict":
		err = predict(docs, *model, *question, *int8, *bias, *out)
	default:
		err = fmt.Errorf("unknown command %q", os.Args[1])
	}
	if err != nil {
		log.Fatal(err)
	}
}

func read(path string) ([]document, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var docs []document
	return docs, json.Unmarshal(b, &docs)
}

// byteOffsets maps each character index of text (and the end) to its
// byte offset.
func byteOffsets(text string) []int {
	offs := make([]int, 0, len(text)+1)
	for i := range text {
		offs = append(offs, i)
	}
	return append(offs, len(text))
}

func convert(docs []document, out string) error {
	var ex []dataset.Example
	dropped, kept := 0, 0
	for _, d := range docs {
		names := make([]string, 0, len(d.Annotations))
		for name := range d.Annotations {
			names = append(names, name)
		}
		sort.Strings(names)
		ms := d.Annotations[names[0]].Mentions
		sort.Slice(ms, func(a, b int) bool { return ms[a].Start < ms[b].Start })
		bo := byteOffsets(d.Text)
		ents, masks := []any{}, []any{}
		end := -1
		for _, m := range ms {
			if m.Start < 0 || m.End > len(bo)-1 || m.Start >= m.End || m.Start < end {
				dropped++ // overlapping or malformed: BIO cannot hold it
				continue
			}
			end = m.End
			s, e := bo[m.Start], bo[m.End]
			ents = append(ents, map[string]any{"start": s, "end": e, "type": m.Type})
			if m.Identifier == "DIRECT" || m.Identifier == "QUASI" {
				masks = append(masks, map[string]any{"start": s, "end": e, "type": m.Identifier})
			}
			kept++
		}
		ex = append(ex, dataset.Example{Text: d.Text, Labels: map[string]any{"entities": ents, "mask": masks}, Family: d.ID, Split: d.Split})
	}
	log.Printf("%d documents, %d mentions kept, %d overlapping dropped", len(ex), kept, dropped)
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	if err := dataset.WriteJSONL(f, ex); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func predict(docs []document, dir, question string, int8 bool, bias float64, out string) error {
	opts := []indecis.Option{indecis.WithSpanBias(question, bias)}
	if int8 {
		opts = append(opts, indecis.WithInt8())
	}
	m, err := indecis.Load(dir, opts...)
	if err != nil {
		return err
	}
	masked := map[string][][2]int{}
	for i, d := range docs {
		dec, err := m.Decide(context.Background(), d.Text)
		if err != nil {
			return err
		}
		spans := [][2]int{}
		for _, s := range dec[0][question].Spans {
			// bytes to characters
			spans = append(spans, [2]int{utf8.RuneCountInString(d.Text[:s.Start]), utf8.RuneCountInString(d.Text[:s.End])})
		}
		masked[d.ID] = spans
		if (i+1)%20 == 0 {
			log.Printf("%d/%d documents", i+1, len(docs))
		}
	}
	b, err := json.Marshal(masked)
	if err != nil {
		return err
	}
	return os.WriteFile(out, b, 0o644)
}
