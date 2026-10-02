// Command compare runs go-anon's CRF recognizer and an indecis model on the
// same plain texts and scores their entities against the labels.
//
// Both engines read the raw text, as in production, and return byte
// offsets. A found entity counts if its type and its bounds are exact
// (strict); the lenient score only asks for the type and an overlap.
//
//	compare -test data/test_clean.jsonl -crf ../go-anon/models/model_fr_pruned.crf.gz -indecis runs/a8m
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/bornholm/go-anon/pkg/features"
	"github.com/bornholm/go-anon/pkg/ner"
	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

type entity struct {
	start, end int
	typ        string
}

// engine finds the entities of a batch of texts.
type engine struct {
	name string
	find func(texts []string) ([][]entity, error)
}

// types are the types of the labels, set from the test file.
var types []string

func main() {
	testPath := flag.String("test", "data/test_clean.jsonl", "labeled examples")
	question := flag.String("question", "entities", "label holding the entities")
	crfPath := flag.String("crf", "", "go-anon CRF model (.crf.gz), empty: skip")
	preset := flag.String("preset", "", `go-anon preset: "", "balanced" or "high-recall" (needs -firstnames)`)
	firstNames := flag.String("firstnames", "", "first-name gazetteer of the presets (default: the firstnames entry of -gazetteers)")
	gazetteers := flag.String("gazetteers", "", `CRF gazetteers, "name:file,name:file", the ones it was trained with`)
	clusters := flag.String("clusters", "", "CRF Brown clusters, the ones it was trained with")
	models := flag.String("indecis", "", "indecis model directories, comma-separated, empty: skip")
	int8 := flag.Bool("int8", false, "indecis in int8")
	threads := flag.Int("threads", 1, "cores per indecis decision")
	maxLen := flag.Int("maxlen", 0, "indecis window length in tokens (0: the model's)")
	merge := flag.Bool("merge", false, "score passages whatever their type: masked or not")
	bias := flag.Float64("bias", 0, "indecis decoding bias towards passages (indecis.WithSpanBias)")
	jsonOut := flag.String("json", "", "write the scores as JSON to this file")
	showErrors := flag.Int("errors", 0, "print this many errors per engine")
	flag.Parse()

	examples, err := dataset.ReadFile(*testPath)
	if err != nil {
		log.Fatal(err)
	}
	seenTypes := map[string]bool{}
	gold := make([][]entity, len(examples))
	texts := make([]string, len(examples))
	for i, e := range examples {
		texts[i] = e.Text
		list, _ := e.Labels[*question].([]any)
		for _, item := range list {
			m := item.(map[string]any)
			typ := m["type"].(string)
			if *merge {
				typ = "ANY"
			}
			gold[i] = append(gold[i], entity{int(m["start"].(float64)), int(m["end"].(float64)), typ})
			seenTypes[typ] = true
		}
	}
	for t := range seenTypes {
		types = append(types, t)
	}
	sort.Strings(types)

	var engines []engine
	if *crfPath != "" {
		e, err := crfEngine(*crfPath, *preset, *firstNames, *gazetteers, *clusters)
		if err != nil {
			log.Fatal(err)
		}
		engines = append(engines, e)
	}
	for _, dir := range strings.Split(*models, ",") {
		if dir == "" {
			continue
		}
		var opts []indecis.Option
		opts = append(opts, indecis.WithThreads(*threads), indecis.WithSpanBias(*question, *bias))
		if *maxLen > 0 {
			opts = append(opts, indecis.WithMaxLen(*maxLen))
		}
		if *int8 {
			opts = append(opts, indecis.WithInt8())
		}
		m, err := indecis.Load(dir, opts...)
		if err != nil {
			log.Fatal(err)
		}
		name := "indecis " + dir
		if *bias != 0 {
			name += fmt.Sprintf(" bias %g", *bias)
		}
		if *int8 {
			name += " int8"
		}
		engines = append(engines, engine{name, func(texts []string) ([][]entity, error) {
			out := make([][]entity, len(texts))
			for i, t := range texts {
				// One text at a time, as a server would.
				d, err := m.Decide(context.Background(), t)
				if err != nil {
					return nil, err
				}
				for _, s := range d[0][*question].Spans {
					typ := s.Type
					if *merge {
						typ = "ANY"
					}
					out[i] = append(out[i], entity{s.Start, s.End, typ})
				}
			}
			return out, nil
		}})
	}

	chars := 0
	for _, t := range texts {
		chars += len(t)
	}
	var report []result
	for _, e := range engines {
		start := time.Now()
		found, err := e.find(texts)
		if err != nil {
			log.Fatal(err)
		}
		elapsed := time.Since(start)
		r := score(e.name, gold, found)
		r.Seconds = elapsed.Seconds()
		r.MsPerText = 1000 * elapsed.Seconds() / float64(len(texts))
		r.MsPer1kBytes = 1000 * elapsed.Seconds() / (float64(chars) / 1000)
		report = append(report, r)
		r.print()
		if *showErrors > 0 {
			printErrors(texts, gold, found, *showErrors)
		}
	}
	if *jsonOut != "" {
		b, _ := json.MarshalIndent(map[string]any{"test": *testPath, "texts": len(texts), "bytes": chars, "results": report}, "", "  ")
		if err := os.WriteFile(*jsonOut, b, 0o644); err != nil {
			log.Fatal(err)
		}
	}
}

func loadGazetteer(name, path string) (*features.Gazetteer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return features.LoadGazetteer(name, f)
}

func crfEngine(path, preset, firstNamesPath, gazetteers, clustersPath string) (engine, error) {
	f, err := os.Open(path)
	if err != nil {
		return engine{}, err
	}
	defer f.Close()
	m, err := ner.LoadModel(f)
	if err != nil {
		return engine{}, err
	}
	opts := []ner.RecognizerOption{ner.WithLanguage("fr")}
	gaz := map[string]*features.Gazetteer{}
	for _, entry := range strings.Split(gazetteers, ",") {
		name, file, ok := strings.Cut(entry, ":")
		if !ok {
			continue
		}
		if gaz[name], err = loadGazetteer(name, file); err != nil {
			return engine{}, err
		}
	}
	if len(gaz) > 0 {
		opts = append(opts, ner.WithGazetteers(gaz))
	}
	if clustersPath != "" {
		c, err := os.Open(clustersPath)
		if err != nil {
			return engine{}, err
		}
		bc, err := features.LoadBrownClusters(c)
		c.Close()
		if err != nil {
			return engine{}, err
		}
		opts = append(opts, ner.WithBrownClusters(bc))
	}
	if preset != "" {
		first := gaz["firstnames"]
		if firstNamesPath != "" {
			if first, err = loadGazetteer("firstnames", firstNamesPath); err != nil {
				return engine{}, err
			}
		}
		if first == nil {
			return engine{}, fmt.Errorf("preset %s: no first-name gazetteer (-firstnames or -gazetteers firstnames:...)", preset)
		}
		opts = append(opts, ner.PresetOptions(ner.Preset(preset), first)...)
	}
	r, err := ner.New(m, opts...)
	if err != nil {
		return engine{}, err
	}
	for _, w := range r.Warnings() {
		log.Printf("go-anon: %s", w)
	}
	name := "go-anon CRF"
	if preset != "" {
		name += " " + preset
	}
	return engine{name, func(texts []string) ([][]entity, error) {
		out := make([][]entity, len(texts))
		for i, t := range texts {
			ents, err := r.Recognize(t)
			if err != nil {
				return nil, err
			}
			for _, e := range ents {
				out[i] = append(out[i], entity{e.Start, e.End, string(e.Type)})
			}
		}
		return out, nil
	}}, nil
}

type counts struct {
	Type      string  `json:"type"`
	Gold      int     `json:"gold"`
	Found     int     `json:"found"`
	Strict    int     `json:"strict"`
	Lenient   int     `json:"lenient"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
	F2        float64 `json:"f2"`
	// LenientRecall: share of the labeled entities overlapped by a found
	// entity of the same type, what masking cares about.
	LenientRecall float64 `json:"lenient_recall"`
	// LenientPrecision: share of the found entities that overlap a labeled
	// one of the same type.
	LenientPrecision float64 `json:"lenient_precision"`
	overlapping      int
}

type result struct {
	Engine       string   `json:"engine"`
	All          counts   `json:"all"`
	Types        []counts `json:"types"`
	Seconds      float64  `json:"seconds"`
	MsPerText    float64  `json:"ms_per_text"`
	MsPer1kBytes float64  `json:"ms_per_1k_bytes"`
}

func (c *counts) finish() {
	div := func(a, b float64) float64 {
		if b == 0 {
			return 0
		}
		return a / b
	}
	c.Precision = div(float64(c.Strict), float64(c.Found))
	c.Recall = div(float64(c.Strict), float64(c.Gold))
	c.F1 = div(2*c.Precision*c.Recall, c.Precision+c.Recall)
	c.F2 = div(5*c.Precision*c.Recall, 4*c.Precision+c.Recall)
	c.LenientRecall = div(float64(c.Lenient), float64(c.Gold))
	c.LenientPrecision = div(float64(c.overlapping), float64(c.Found))
}

func score(name string, gold, found [][]entity) result {
	per := map[string]*counts{}
	for _, t := range types {
		per[t] = &counts{Type: t}
	}
	all := counts{}
	for i := range gold {
		exact := map[entity]bool{}
		for _, g := range gold[i] {
			exact[g] = true
			if c := per[g.typ]; c != nil {
				c.Gold++
			}
			all.Gold++
			for _, f := range found[i] {
				if f.typ == g.typ && f.start < g.end && g.start < f.end {
					if c := per[g.typ]; c != nil {
						c.Lenient++
					}
					all.Lenient++
					break
				}
			}
		}
		for _, f := range found[i] {
			c := per[f.typ]
			if c == nil {
				continue // types the labels do not have (regex identifiers)
			}
			c.Found++
			all.Found++
			for _, g := range gold[i] {
				if f.typ == g.typ && f.start < g.end && g.start < f.end {
					c.overlapping++
					all.overlapping++
					break
				}
			}
			if exact[f] {
				c.Strict++
				all.Strict++
			}
		}
	}
	r := result{Engine: name}
	all.finish()
	r.All = all
	for _, t := range types {
		per[t].finish()
		r.Types = append(r.Types, *per[t])
	}
	return r
}

func (r result) print() {
	fmt.Printf("\n%s  (%.1f s, %.1f ms/text, %.2f ms/kB)\n", r.Engine, r.Seconds, r.MsPerText, r.MsPer1kBytes)
	fmt.Printf("  %-5s %6s %6s %6s %6s %6s %6s %8s %8s\n", "type", "gold", "found", "P", "R", "F1", "F2", "R(lenient)", "P(lenient)")
	for _, c := range append(r.Types, r.All) {
		t := c.Type
		if t == "" {
			t = "all"
		}
		fmt.Printf("  %-5s %6d %6d %6.3f %6.3f %6.3f %6.3f %8.3f %8.3f\n", t, c.Gold, c.Found, c.Precision, c.Recall, c.F1, c.F2, c.LenientRecall, c.LenientPrecision)
	}
}

func printErrors(texts []string, gold, found [][]entity, n int) {
	shown := 0
	for i := range texts {
		g := map[entity]bool{}
		for _, e := range gold[i] {
			g[e] = true
		}
		f := map[entity]bool{}
		for _, e := range found[i] {
			f[e] = true
		}
		var missed, extra []string
		for _, e := range gold[i] {
			if !f[e] {
				missed = append(missed, fmt.Sprintf("%s %q", e.typ, texts[i][e.start:e.end]))
			}
		}
		for _, e := range found[i] {
			if !g[e] && e.end <= len(texts[i]) {
				extra = append(extra, fmt.Sprintf("%s %q", e.typ, texts[i][e.start:e.end]))
			}
		}
		if len(missed)+len(extra) == 0 {
			continue
		}
		sort.Strings(missed)
		sort.Strings(extra)
		fmt.Printf("  - %q\n    missed %v\n    extra  %v\n", texts[i], missed, extra)
		if shown++; shown >= n {
			return
		}
	}
}
