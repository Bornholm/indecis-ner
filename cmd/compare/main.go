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
	"slices"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/bornholm/go-anon/pkg/features"
	"github.com/bornholm/go-anon/pkg/ner"
	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis-ner/lexicon"
	"github.com/bornholm/indecis-ner/recognizer"
	"github.com/bornholm/indecis/dataset"
)

type entity struct {
	start, end int
	typ        string
}

// engine finds the entities of a batch of texts.
type engine struct {
	name string
	find func(texts []string) (findings, error)
}

// findings are what an engine found in each text: the entities and, for
// indecis with -refine and the sensitive question, the refined masks that
// carry a special category and the model's sensitive passages before
// refining.
type findings struct {
	found, flagged, raw [][]entity
}

// types are the types of the labels, set from the test file.
var types []string

func main() {
	testPath := flag.String("test", "data/test_clean.jsonl", "labeled examples")
	question := flag.String("question", "entities", "label holding the entities; several, comma-separated, are scored as their union (mask,sensitive)")
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
	novelFlag := flag.String("novel", "", "lexicon of the training set (cmd/lexicon): score only the labeled passages none of whose words the training set had in their category; found passages over the others are ignored")
	bias := flag.Float64("bias", 0, "indecis decoding bias towards passages (indecis.WithSpanBias)")
	minConfidence := flag.Float64("minconfidence", 0, "drop the indecis passages less confident, sensitive ones included, as the recognizer's MinConfidence")
	jsonOut := flag.String("json", "", "write the scores as JSON to this file")
	showErrors := flag.Int("errors", 0, "print this many errors per engine")
	showLeaks := flag.Int("leaks", 0, "print this many labeled passages left partly or fully unmasked")
	showExtra := flag.Int("extra", 0, "print this many found passages that overlap no labeled one")
	refine := flag.Bool("refine", false, "refine indecis masks as the go-anon adapter does (recognizer.Masks): grow the masks cut inside a word to the whole word and leave out the spaces, the punctuation of sentences and the unpaired brackets at their edges, cut a mask that holds several names or identifiers, drop function-word masks, join masks over function words when they hold the same special category, or none; with the sensitive question, also report the categories the masks keep; implies -merge")
	flag.Parse()
	novel := novelFlag
	if *refine {
		*merge = true
	}

	examples, err := dataset.ReadFile(*testPath)
	if err != nil {
		log.Fatal(err)
	}
	var lex lexicon.Lexicon
	if *novel != "" {
		if lex, err = lexicon.Load(*novel); err != nil {
			log.Fatal(err)
		}
	}
	seenTypes := map[string]bool{}
	gold := make([][]entity, len(examples))
	// sensitive holds the labeled special categories, for the categories
	// the refined masks keep: with -refine and the sensitive question.
	sensitive := make([][]entity, len(examples))
	questions := strings.Split(*question, ",")
	categories := *refine && slices.Contains(questions, "sensitive")
	ignored := make([][]entity, len(examples)) // labeled but not novel
	kept, dropped := 0, 0
	texts := make([]string, len(examples))
	for i, e := range examples {
		texts[i] = e.Text
		if categories {
			labels, _ := e.Labels["sensitive"].([]any)
			seen := map[entity]bool{}
			for _, item := range labels {
				ent, err := label(e.Text, item)
				if err != nil {
					log.Fatalf("%s, example %d, sensitive: %v", *testPath, i+1, err)
				}
				// With -novel, as the other labels.
				if seen[ent] || lex != nil && !lex.Novel(ent.typ, e.Text[ent.start:ent.end]) {
					continue
				}
				seen[ent] = true
				sensitive[i] = append(sensitive[i], ent)
			}
		}
		seen := map[entity]bool{}
		for _, q := range questions {
			l, _ := e.Labels[q].([]any)
			for _, item := range l {
				ent, err := label(e.Text, item)
				if err != nil {
					log.Fatalf("%s, example %d, %s: %v", *testPath, i+1, q, err)
				}
				typ := ent.typ
				if *merge {
					ent.typ = "ANY"
				}
				if lex != nil && !lex.Novel(typ, e.Text[ent.start:ent.end]) {
					ignored[i] = append(ignored[i], ent)
					dropped++
					continue
				}
				if seen[ent] {
					continue // the same passage under two questions
				}
				seen[ent] = true
				kept++
				gold[i] = append(gold[i], ent)
				seenTypes[ent.typ] = true
			}
		}
	}
	if lex != nil {
		log.Printf("%d novel labeled passages scored, %d ignored (words seen in training)", kept, dropped)
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
		opts = append(opts, indecis.WithThreads(*threads))
		for _, q := range questions {
			opts = append(opts, indecis.WithSpanBias(q, *bias))
		}
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
		if *refine {
			name += " refined"
		}
		if *bias != 0 {
			name += fmt.Sprintf(" bias %g", *bias)
		}
		if *int8 {
			name += " int8"
		}
		engines = append(engines, engine{name, func(texts []string) (findings, error) {
			out := make([][]entity, len(texts))
			var flagged, raw [][]entity
			if categories {
				flagged, raw = make([][]entity, len(texts)), make([][]entity, len(texts))
			}
			// As the recognizer: the passages under -minconfidence, the
			// sensitive ones included, are dropped before refining.
			confident := func(spans []indecis.Span) []indecis.Span {
				kept := spans[:0:0]
				for _, s := range spans {
					if s.Confidence >= *minConfidence {
						kept = append(kept, s)
					}
				}
				return kept
			}
			for i, t := range texts {
				// One text at a time, as a server would.
				d, err := m.Decide(context.Background(), t)
				if err != nil {
					return findings{}, err
				}
				var spans []indecis.Span
				for _, q := range questions {
					// As the recognizer: a question the model lacks is an
					// error, not a passage-free answer.
					a, ok := d[0][q]
					if !ok || a.Kind != indecis.Spans {
						return findings{}, fmt.Errorf("%s has no spans question %s", dir, q)
					}
					for _, s := range confident(a.Spans) {
						typ := s.Type
						if *merge {
							typ = "ANY"
						}
						if *refine {
							spans = append(spans, s)
							continue
						}
						out[i] = append(out[i], entity{s.Start, s.End, typ})
					}
				}
				if *refine {
					// As the recognizer: the sensitive passages count only
					// when the question is asked.
					var sens []indecis.Span
					if categories {
						sens = confident(d[0]["sensitive"].Spans)
					}
					for _, s := range sens {
						raw[i] = append(raw[i], entity{s.Start, s.End, s.Type})
					}
					for _, m := range recognizer.Masks(t, spans, d[0]["entities"].Spans, sens) {
						out[i] = append(out[i], entity{m.Start, m.End, "ANY"})
						if categories && m.Sensitive != "" {
							flagged[i] = append(flagged[i], entity{m.Start, m.End, m.Sensitive})
						}
					}
				}
			}
			return findings{out, flagged, raw}, nil
		}})
	}

	chars := 0
	for _, t := range texts {
		chars += len(t)
	}
	var report []result
	for _, e := range engines {
		start := time.Now()
		f, err := e.find(texts)
		if err != nil {
			log.Fatal(err)
		}
		found := withoutIgnored(f.found, ignored)
		elapsed := time.Since(start)
		r := score(e.name, gold, found)
		r.Chars = leakage(texts, gold, found, ignored)
		r.Seconds = elapsed.Seconds()
		r.MsPerText = 1000 * elapsed.Seconds() / float64(len(texts))
		r.MsPer1kBytes = 1000 * elapsed.Seconds() / (float64(chars) / 1000)
		if f.flagged != nil {
			// With -novel, the masks over ignored passages are left out, as
			// from found.
			c := countCategories(texts, sensitive, found, outsideIgnored(f.flagged, ignored), outsideIgnored(f.raw, ignored))
			r.Categories = &c
		}
		report = append(report, r)
		r.print()
		if r.Categories != nil {
			r.Categories.print()
		} else if categories {
			fmt.Println("  special categories: not reported, the engine gives none")
		}
		if *showErrors > 0 {
			printErrors(texts, gold, found, *showErrors)
		}
		if *showLeaks > 0 {
			printLeaks(texts, examples, questions, found, *showLeaks)
		}
		if *showExtra > 0 {
			printExtra(texts, gold, found, *showExtra)
		}
	}
	if *jsonOut != "" {
		b, _ := json.MarshalIndent(map[string]any{"test": *testPath, "texts": len(texts), "bytes": chars, "results": report}, "", "  ")
		if err := os.WriteFile(*jsonOut, b, 0o644); err != nil {
			log.Fatal(err)
		}
	}
}

// label reads a labeled passage of text, and fails on one malformed or
// out of the text.
func label(text string, item any) (entity, error) {
	m, _ := item.(map[string]any)
	start, ok1 := m["start"].(float64)
	end, ok2 := m["end"].(float64)
	typ, ok3 := m["type"].(string)
	if !ok1 || !ok2 || !ok3 || start < 0 || start >= end || int(end) > len(text) {
		return entity{}, fmt.Errorf("malformed label %v", item)
	}
	return entity{int(start), int(end), typ}, nil
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
	return engine{name, func(texts []string) (findings, error) {
		out := make([][]entity, len(texts))
		for i, t := range texts {
			ents, err := r.Recognize(t)
			if err != nil {
				return findings{}, err
			}
			for _, e := range ents {
				out[i] = append(out[i], entity{e.Start, e.End, string(e.Type)})
			}
		}
		return findings{found: out}, nil
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
	Chars        chars    `json:"chars"`
	// Categories, with -refine and the sensitive question.
	Categories *categories `json:"categories,omitempty"`
}

// chars measures masking character by character, whatever the types:
// spaces and punctuation are not counted, as in TAB's token measures.
type chars struct {
	Labeled int `json:"labeled"` // characters of labeled passages
	Leaked  int `json:"leaked"`  // of those, left unmasked
	Masked  int `json:"masked"`  // characters masked
	// Right is the number of masked characters that belong to a labeled
	// passage.
	Right int `json:"right"`
	// CleanDocs is the number of documents with no labeled character left
	// unmasked.
	CleanDocs int `json:"clean_docs"`
	Docs      int `json:"docs"`
	// Exposed is the number of labeled passages of which less than half
	// the counted characters are masked, out of Passages.
	Exposed  int `json:"exposed"`
	Passages int `json:"passages"`
}

// functionWords carry no information on their own: like TAB's measures,
// which leave out determiners and prepositions, the leak counts don't
// include them.
var functionWords = map[string]bool{}

func init() {
	for _, w := range strings.Fields(`le la les l un une des de du d au aux a à en sur dans par pour avec chez sous et ou mais son sa ses mon ma mes ton ta tes notre nos votre vos leur leurs ce cet cette ces qui que qu il elle ils elles on se s y the of a an in on at to and`) {
		functionWords[w] = true
	}
}

// markFunctionWords flags the bytes of the function words of a text.
func markFunctionWords(t string, skip []bool) {
	start := -1
	for k, r := range t + " " {
		letter := unicode.IsLetter(r)
		if letter && start < 0 {
			start = k
		}
		if !letter && start >= 0 {
			if functionWords[strings.ToLower(t[start:k])] {
				for j := start; j < k; j++ {
					skip[j] = true
				}
			}
			start = -1
		}
	}
}

// leakage compares the characters covered by the labeled passages with
// those covered by the found ones. Ignored passages (see -novel) and
// function words count on neither side.
func leakage(texts []string, gold, found, ignored [][]entity) chars {
	var c chars
	for i, t := range texts {
		g := make([]bool, len(t))
		f := make([]bool, len(t))
		skip := make([]bool, len(t))
		markFunctionWords(t, skip)
		mark := func(m []bool, es []entity) {
			for _, e := range es {
				for k := max(e.start, 0); k < min(e.end, len(t)); k++ {
					m[k] = true
				}
			}
		}
		mark(g, gold[i])
		mark(f, found[i])
		mark(skip, ignored[i])
		leaked := 0
		for k, r := range t { // k: start of each character
			if skip[k] || unicode.IsSpace(r) || unicode.IsPunct(r) {
				continue
			}
			if g[k] {
				c.Labeled++
				if !f[k] {
					c.Leaked++
					leaked++
				}
			}
			if f[k] {
				c.Masked++
				if g[k] {
					c.Right++
				}
			}
		}
		c.Docs++
		if leaked == 0 {
			c.CleanDocs++
		}
		for _, e := range gold[i] {
			n, covered := 0, 0
			for k, r := range t[max(e.start, 0):min(e.end, len(t))] {
				k += e.start
				if skip[k] || unicode.IsSpace(r) || unicode.IsPunct(r) {
					continue
				}
				n++
				if f[k] {
					covered++
				}
			}
			if n == 0 {
				continue
			}
			c.Passages++
			if 2*covered < n {
				c.Exposed++
			}
		}
	}
	return c
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
	ch := r.Chars
	if ch.Labeled > 0 && ch.Masked > 0 {
		fmt.Printf("  characters: %.1f%% of the labeled ones leaked (%d of %d), %.1f%% of the masked ones labeled, %d of %d documents without a leak\n",
			100*float64(ch.Leaked)/float64(ch.Labeled), ch.Leaked, ch.Labeled, 100*float64(ch.Right)/float64(ch.Masked), ch.CleanDocs, ch.Docs)
		fmt.Printf("  passages: %d of %d exposed (%.1f%%, less than half masked)\n", ch.Exposed, ch.Passages, 100*float64(ch.Exposed)/float64(max(ch.Passages, 1)))
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

// withoutIgnored drops the found passages that overlap an ignored labeled
// passage of the same type: they are neither right nor wrong for the
// novel subset.
func withoutIgnored(found, ignored [][]entity) [][]entity {
	out := make([][]entity, len(found))
	for i, fs := range found {
	next:
		for _, f := range fs {
			for _, g := range ignored[i] {
				if f.typ == g.typ && f.start < g.end && g.start < f.end {
					continue next
				}
			}
			out[i] = append(out[i], f)
		}
	}
	return out
}

// outsideIgnored drops the passages that overlap an ignored labeled
// passage, whatever their type: the flagged masks carry a category, not
// the type of the labels.
func outsideIgnored(list, ignored [][]entity) [][]entity {
	out := make([][]entity, len(list))
	for i, fs := range list {
		for _, f := range fs {
			if !slices.ContainsFunc(ignored[i], func(g entity) bool { return f.start < g.end && g.start < f.end }) {
				out[i] = append(out[i], f)
			}
		}
	}
	return out
}

// printLeaks prints labeled passages with characters left unmasked, the
// unmasked part in brackets, with their labels under every question.
func printLeaks(texts []string, examples []dataset.Example, questions []string, found [][]entity, n int) {
	shown := 0
	for i, t := range texts {
		covered := make([]bool, len(t))
		for _, f := range found[i] {
			for k := max(f.start, 0); k < min(f.end, len(t)); k++ {
				covered[k] = true
			}
		}
		for _, q := range questions {
			list, _ := examples[i].Labels[q].([]any)
			for _, item := range list {
				m := item.(map[string]any)
				s, e := int(m["start"].(float64)), int(m["end"].(float64))
				var b strings.Builder
				leak, open := false, false
				for k, r := range t[s:e] {
					in := !covered[s+k] && !unicode.IsSpace(r) && !unicode.IsPunct(r)
					if in && !open {
						b.WriteString("[")
						open = true
					} else if !in && open && !unicode.IsSpace(r) {
						b.WriteString("]")
						open = false
					}
					leak = leak || in
					b.WriteRune(r)
				}
				if open {
					b.WriteString("]")
				}
				if leak {
					fmt.Printf("  %-9s %-9s %s\n", q, m["type"], b.String())
					if shown++; shown >= n {
						return
					}
				}
			}
		}
	}
}

// printExtra prints found passages that overlap no labeled passage, with
// some context.
func printExtra(texts []string, gold, found [][]entity, n int) {
	shown := 0
	for i, t := range texts {
	next:
		for _, f := range found[i] {
			for _, g := range gold[i] {
				if f.start < g.end && g.start < f.end {
					continue next
				}
			}
			a, b := max(f.start-40, 0), min(f.end+40, len(t))
			for a > 0 && !utf8.RuneStart(t[a]) {
				a--
			}
			for b < len(t) && !utf8.RuneStart(t[b]) {
				b++
			}
			ctx := strings.ReplaceAll(t[a:f.start]+"[["+t[f.start:f.end]+"]]"+t[f.end:b], "\n", " ")
			fmt.Printf("  %s\n", ctx)
			if shown++; shown >= n {
				return
			}
		}
	}
}

// categories tells whether the refined masks keep the special category
// of the labeled sensitive passages they mask: a mask without it is
// treated by go-anon's users as any personal data. A passage counts as
// masked, or flagged, when masks, or masks with a category, cover at
// least half of its letters and digits. Flagged masks that overlap no
// labeled sensitive passage are the cost. The model's own sensitive
// passages, before refining, give the ceiling.
type categories struct {
	Labeled int `json:"labeled"` // labeled sensitive passages
	Masked  int `json:"masked"`  // of those, masked
	// Flagged: of the masked ones, those whose masks carry a category,
	// any; Kept: the labeled one.
	Flagged int `json:"flagged"`
	Kept    int `json:"kept"`
	// Ceiling: of the masked ones, those the model's sensitive passages
	// cover before refining.
	Ceiling int `json:"ceiling"`
	// Masks flagged sensitive, and Stray those over no labeled sensitive
	// passage.
	Masks int `json:"masks"`
	Stray int `json:"stray"`
}

func countCategories(texts []string, sensitive, found, flagged, raw [][]entity) categories {
	covered := func(t string, g entity, list []entity, typ string) (n, of int) {
		for i, r := range t[g.start:g.end] {
			if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
				continue
			}
			of++
			for _, e := range list {
				if e.start <= g.start+i && g.start+i < e.end && (typ == "" || e.typ == typ) {
					n++
					break
				}
			}
		}
		return n, of
	}
	var c categories
	for i, t := range texts {
		for _, g := range sensitive[i] {
			c.Labeled++
			if n, of := covered(t, g, found[i], ""); of == 0 || 2*n < of {
				continue
			}
			c.Masked++
			if n, of := covered(t, g, raw[i], ""); 2*n >= of {
				c.Ceiling++
			}
			if n, of := covered(t, g, flagged[i], ""); 2*n >= of {
				c.Flagged++
			}
			if n, of := covered(t, g, flagged[i], g.typ); 2*n >= of {
				c.Kept++
			}
		}
		for _, f := range flagged[i] {
			c.Masks++
			if !slices.ContainsFunc(sensitive[i], func(g entity) bool { return g.start < f.end && f.start < g.end }) {
				c.Stray++
			}
		}
	}
	return c
}

func (c categories) print() {
	pct := func(a, b int) float64 { return 100 * float64(a) / float64(max(b, 1)) }
	fmt.Printf("  special categories: %d labeled passages, %d masked; their mask flagged sensitive: %d (%.1f%%), with their category: %d (%.1f%%)\n",
		c.Labeled, c.Masked, c.Flagged, pct(c.Flagged, c.Masked), c.Kept, pct(c.Kept, c.Masked))
	fmt.Printf("  the model's sensitive passages, before refining, cover %d of them (%.1f%%)\n", c.Ceiling, pct(c.Ceiling, c.Masked))
	fmt.Printf("  masks flagged sensitive: %d, over no labeled sensitive passage: %d (%.1f%%)\n", c.Masks, c.Stray, pct(c.Stray, c.Masks))
}
