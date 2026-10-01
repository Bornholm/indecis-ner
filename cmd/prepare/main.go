// Command prepare turns go-anon's WikiNER corpus into indecis examples:
// sentences rebuilt as plain text, entities as byte offsets.
//
// WikiNER repeats many sentences, and 85% of go-anon's French test
// sentences also appear in its training set. prepare therefore writes:
//
//   - train.jsonl: the distinct training sentences, without any test or dev
//     sentence;
//   - dev.jsonl: the dev sentences absent from train and test;
//   - test_clean.jsonl: the test sentences absent from train and dev, the
//     only fair test for the CRF, which was trained on train + dev;
//   - test_full.jsonl: the whole test set, to compare with the scores go-anon
//     publishes;
//   - docs.jsonl: clean test sentences joined into documents of about
//     -doc-words words, to measure long texts.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strings"

	"github.com/bornholm/indecis/dataset"
)

type token struct{ word, tag string }

type sentence []token

func (s sentence) key() string {
	words := make([]string, len(s))
	for i, t := range s {
		words[i] = t.word
	}
	return strings.Join(words, " ")
}

func main() {
	dir := flag.String("data", "../go-anon/data/fr", "go-anon data directory")
	out := flag.String("out", "data", "output directory")
	docWords := flag.Int("doc-words", 1500, "approximate length of the documents of docs.jsonl, in words")
	seed := flag.Int64("seed", 1, "seed of the document assembly")
	flag.Parse()

	train, err := readWikiNER(filepath.Join(*dir, "wikiner_fr_full.train.wikiner"))
	if err != nil {
		log.Fatal(err)
	}
	dev, err := readWikiNER(filepath.Join(*dir, "wikiner_fr_full.dev.wikiner"))
	if err != nil {
		log.Fatal(err)
	}
	test, err := readCoNLL(filepath.Join(*dir, "wikiner_fr.test.conll"))
	if err != nil {
		log.Fatal(err)
	}
	keys := func(ss []sentence) map[string]bool {
		m := map[string]bool{}
		for _, s := range ss {
			m[s.key()] = true
		}
		return m
	}
	trainKeys, devKeys, testKeys := keys(train), keys(dev), keys(test)

	distinct := func(ss []sentence, exclude ...map[string]bool) []sentence {
		seen := map[string]bool{}
		var kept []sentence
	next:
		for _, s := range ss {
			k := s.key()
			if seen[k] {
				continue
			}
			seen[k] = true
			for _, ex := range exclude {
				if ex[k] {
					continue next
				}
			}
			kept = append(kept, s)
		}
		return kept
	}
	sets := []struct {
		name string
		ss   []sentence
	}{
		{"train", distinct(train, devKeys, testKeys)},
		{"dev", distinct(dev, trainKeys, testKeys)},
		{"test_clean", distinct(test, trainKeys, devKeys)},
		{"test_full", test},
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}
	for _, set := range sets {
		var ex []dataset.Example
		for _, s := range set.ss {
			ex = append(ex, toExample(s, set.name))
		}
		if err := write(filepath.Join(*out, set.name+".jsonl"), ex); err != nil {
			log.Fatal(err)
		}
		log.Printf("%s: %d sentences (%d read)", set.name, len(ex), len(set.ss))
	}
	log.Printf("raw: train %d, dev %d, test %d sentences", len(train), len(dev), len(test))

	clean := sets[2].ss
	rng := rand.New(rand.NewSource(*seed))
	order := rng.Perm(len(clean))
	var docs []dataset.Example
	var cur []sentence
	words := 0
	flush := func() {
		if len(cur) > 0 {
			docs = append(docs, joinDoc(cur, fmt.Sprintf("doc%03d", len(docs))))
		}
		cur, words = nil, 0
	}
	for _, i := range order {
		cur = append(cur, clean[i])
		words += len(clean[i])
		if words >= *docWords {
			flush()
		}
	}
	flush()
	if err := write(filepath.Join(*out, "docs.jsonl"), docs); err != nil {
		log.Fatal(err)
	}
	log.Printf("docs: %d documents of about %d words", len(docs), *docWords)
}

// readWikiNER reads one sentence per line, tokens word|POS|tag, tags in
// IO: consecutive tokens of the same type form one entity.
func readWikiNER(path string) ([]sentence, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []sentence
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var s sentence
		prev := "O"
		for _, field := range strings.Fields(line) {
			parts := strings.Split(field, "|")
			if len(parts) < 3 {
				return nil, fmt.Errorf("%s: token %q", path, field)
			}
			word, tag := strings.Join(parts[:len(parts)-2], "|"), parts[len(parts)-1]
			// IO to BIO: an I- after O or another type begins an entity.
			if strings.HasPrefix(tag, "I-") && prev[min(2, len(prev)):] != tag[2:] {
				tag = "B-" + tag[2:]
			}
			prev = tag
			s = append(s, token{word, tag})
		}
		out = append(out, s)
	}
	return out, sc.Err()
}

// readCoNLL reads one token per line (word, POS, BIO tag), sentences
// separated by blank lines.
func readCoNLL(path string) ([]sentence, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []sentence
	var cur sentence
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimRight(sc.Text(), "\r\n")
		if strings.TrimSpace(line) == "" {
			if len(cur) > 0 {
				out = append(out, cur)
				cur = nil
			}
			continue
		}
		cols := strings.Split(line, "\t")
		cur = append(cur, token{cols[0], cols[len(cols)-1]})
	}
	if len(cur) > 0 {
		out = append(out, cur)
	}
	return out, sc.Err()
}

// detok rebuilds the text of a sentence with French typography and
// returns the byte span of each token.
func detok(s sentence) (string, [][2]int) {
	var b strings.Builder
	spans := make([][2]int, len(s))
	for i, t := range s {
		if i > 0 && spaceBefore(s[i-1].word, t.word) {
			b.WriteByte(' ')
		}
		spans[i][0] = b.Len()
		b.WriteString(t.word)
		spans[i][1] = b.Len()
	}
	return b.String(), spans
}

func spaceBefore(prev, word string) bool {
	switch {
	case strings.HasSuffix(prev, "'") || strings.HasSuffix(prev, "’"):
		return false // l', d', qu'
	case prev == "(" || prev == "[":
		return false
	}
	switch word {
	case ",", ".", ")", "]", "...", "…":
		return false
	}
	return true
}

// entities converts BIO tags into entities over the token spans.
func entities(s sentence, spans [][2]int) []any {
	out := []any{}
	for i := 0; i < len(s); {
		tag := s[i].tag
		if tag == "O" || len(tag) < 3 {
			i++
			continue
		}
		typ := tag[2:]
		j := i + 1
		for j < len(s) && s[j].tag == "I-"+typ {
			j++
		}
		out = append(out, map[string]any{"start": spans[i][0], "end": spans[j-1][1], "type": typ})
		i = j
	}
	return out
}

func toExample(s sentence, split string) dataset.Example {
	text, spans := detok(s)
	return dataset.Example{Text: text, Labels: map[string]any{"entities": entities(s, spans)}, Split: split}
}

// joinDoc joins sentences into one document, shifting their entities.
func joinDoc(ss []sentence, id string) dataset.Example {
	var b strings.Builder
	ents := []any{}
	for i, s := range ss {
		if i > 0 {
			b.WriteByte(' ')
		}
		base := b.Len()
		text, spans := detok(s)
		b.WriteString(text)
		for _, e := range entities(s, spans) {
			m := e.(map[string]any)
			m["start"] = m["start"].(int) + base
			m["end"] = m["end"].(int) + base
			ents = append(ents, m)
		}
	}
	return dataset.Example{Text: b.String(), Labels: map[string]any{"entities": ents}, Family: id, Split: "docs"}
}

func write(path string, ex []dataset.Example) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := dataset.WriteJSONL(f, ex); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
