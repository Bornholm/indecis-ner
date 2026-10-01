// Command translate translates TAB documents into French with Claude,
// keeping their annotations: each mention of the first annotator is
// wrapped in numbered markers, ⟦12⟧…⟦/12⟧, that the translation must keep
// around the translated mention. The markers give back the offsets, so the
// French documents carry the same masking decisions without being
// annotated again.
//
//	translate -in data/tab/echr_test.json -out data/tab/echr_test_fr.json -orig data/tab/echr_test_a1.json
//
// -orig writes the English documents reduced to the same annotator, the
// reference to compare the French scores with. Responses are cached in
// -cache: an interrupted run resumes where it stopped.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

type mention map[string]any

type document map[string]any

const systemPrompt = `You translate rulings of the European Court of Human Rights from English into French, as the Court's own French versions are written: same legal register, same structure, paragraph numbers kept.

The text contains markers ⟦n⟧ and ⟦/n⟧ around passages. Rules:
- Translate the passage between ⟦n⟧ and ⟦/n⟧ and keep the two markers around its translation, with the same number.
- Every marker pair of the input appears exactly once in the output, in the order French requires. Never nest, merge, split or drop a pair, never add one.
- Keep names of people, case numbers, codes and addresses as they are. Translate names of institutions, countries, dates, professions and other descriptions.
- Output only the translated text, with its markers, without comments.`

var markerRe = regexp.MustCompile(`⟦(/?)(\d+)⟧`)

func main() {
	in := flag.String("in", "data/tab/echr_test.json", "TAB documents")
	out := flag.String("out", "data/tab/echr_test_fr.json", "French documents, TAB format")
	orig := flag.String("orig", "data/tab/echr_test_a1.json", "English documents reduced to the first annotator, TAB format")
	cacheDir := flag.String("cache", "data/tab/translate_cache", "response cache")
	model := flag.String("model", "sonnet", "Claude model")
	concurrency := flag.Int("concurrency", 2, "simultaneous calls")
	interval := flag.Duration("interval", 5*time.Second, "minimum time between two calls")
	limit := flag.Int("limit", 0, "translate only the first documents (0: all)")
	flag.Parse()

	b, err := os.ReadFile(*in)
	if err != nil {
		log.Fatal(err)
	}
	var docs []document
	if err := json.Unmarshal(b, &docs); err != nil {
		log.Fatal(err)
	}
	if *limit > 0 && *limit < len(docs) {
		docs = docs[:*limit]
	}
	if err := os.MkdirAll(*cacheDir, 0o755); err != nil {
		log.Fatal(err)
	}

	type job struct {
		i        int
		marked   string
		mentions []mention
	}
	var jobs []job
	origDocs := make([]document, len(docs))
	for i, d := range docs {
		marked, ms := markUp(d)
		jobs = append(jobs, job{i, marked, ms})
		origDocs[i] = withMentions(d, d["text"].(string), ms)
	}

	results := make([]document, len(docs))
	var mu sync.Mutex
	var failed []string
	tick := time.NewTicker(*interval)
	defer tick.Stop()
	sem := make(chan struct{}, *concurrency)
	var wg sync.WaitGroup
	for _, j := range jobs {
		id := docs[j.i]["doc_id"].(string)
		cache := filepath.Join(*cacheDir, id+".txt")
		wg.Add(1)
		go func() {
			defer wg.Done()
			var text string
			var err error
			for attempt := 0; attempt < 3; attempt++ {
				resp, cached := readCache(cache)
				if !cached {
					sem <- struct{}{}
					<-tick.C
					resp, err = callClaude(*model, j.marked)
					<-sem
					if err != nil {
						log.Printf("%s: %v", id, err)
						continue
					}
				}
				var ms []mention
				text, ms, err = parse(resp, j.mentions)
				if err != nil {
					log.Printf("%s: attempt %d: %v", id, attempt+1, err)
					os.Remove(cache)
					continue
				}
				if !cached {
					os.WriteFile(cache, []byte(resp), 0o644)
				}
				mu.Lock()
				results[j.i] = withMentions(docs[j.i], text, ms)
				mu.Unlock()
				log.Printf("%s: %d characters, %d mentions", id, utf8.RuneCountInString(text), len(ms))
				return
			}
			mu.Lock()
			failed = append(failed, id)
			mu.Unlock()
		}()
	}
	wg.Wait()

	var done []document
	var doneOrig []document
	for i, r := range results {
		if r != nil {
			done = append(done, r)
			doneOrig = append(doneOrig, origDocs[i])
		}
	}
	if err := writeJSON(*out, done); err != nil {
		log.Fatal(err)
	}
	if err := writeJSON(*orig, doneOrig); err != nil {
		log.Fatal(err)
	}
	log.Printf("%d documents translated, %d failed %v", len(done), len(failed), failed)
}

// markUp wraps the non-overlapping mentions of the first annotator in
// markers numbered by their position.
func markUp(d document) (string, []mention) {
	text := []rune(d["text"].(string))
	anns := d["annotations"].(map[string]any)
	names := make([]string, 0, len(anns))
	for n := range anns {
		names = append(names, n)
	}
	sort.Strings(names)
	raw := anns[names[0]].(map[string]any)["entity_mentions"].([]any)
	var ms []mention
	for _, r := range raw {
		ms = append(ms, mention(r.(map[string]any)))
	}
	sort.SliceStable(ms, func(a, b int) bool { return num(ms[a]["start_offset"]) < num(ms[b]["start_offset"]) })
	var kept []mention
	end := -1
	for _, m := range ms {
		s, e := num(m["start_offset"]), num(m["end_offset"])
		if s < end || s >= e || e > len(text) {
			continue
		}
		end = e
		kept = append(kept, m)
	}
	var b strings.Builder
	pos := 0
	for i, m := range kept {
		s, e := num(m["start_offset"]), num(m["end_offset"])
		b.WriteString(string(text[pos:s]))
		fmt.Fprintf(&b, "⟦%d⟧%s⟦/%d⟧", i, string(text[s:e]), i)
		pos = e
	}
	b.WriteString(string(text[pos:]))
	return b.String(), kept
}

func num(v any) int { return int(v.(float64)) }

// parse removes the markers from a translation and returns the text and
// the mentions at their new character offsets.
func parse(resp string, ms []mention) (string, []mention, error) {
	resp = strings.TrimSpace(resp)
	var b strings.Builder
	chars := 0
	open := -1
	start := map[int]int{}
	endOf := map[int]int{}
	last := 0
	for _, loc := range markerRe.FindAllStringSubmatchIndex(resp, -1) {
		seg := resp[last:loc[0]]
		b.WriteString(seg)
		chars += utf8.RuneCountInString(seg)
		last = loc[1]
		n, _ := strconv.Atoi(resp[loc[4]:loc[5]])
		closing := loc[3] > loc[2]
		switch {
		case n < 0 || n >= len(ms):
			return "", nil, fmt.Errorf("unknown marker %d", n)
		case !closing && open >= 0:
			return "", nil, fmt.Errorf("marker %d opened inside %d", n, open)
		case !closing:
			if _, dup := start[n]; dup {
				return "", nil, fmt.Errorf("marker %d twice", n)
			}
			start[n], open = chars, n
		case closing && open != n:
			return "", nil, fmt.Errorf("marker /%d closes %d", n, open)
		default:
			endOf[n], open = chars, -1
		}
	}
	b.WriteString(resp[last:])
	if open >= 0 {
		return "", nil, fmt.Errorf("marker %d not closed", open)
	}
	if len(endOf) != len(ms) {
		return "", nil, fmt.Errorf("%d of %d markers", len(endOf), len(ms))
	}
	text := b.String()
	runes := []rune(text)
	out := make([]mention, len(ms))
	for i, m := range ms {
		s, e := start[i], endOf[i]
		if s >= e {
			return "", nil, fmt.Errorf("marker %d empty", i)
		}
		c := mention{}
		for k, v := range m {
			c[k] = v
		}
		c["start_offset"], c["end_offset"], c["span_text"] = s, e, string(runes[s:e])
		out[i] = c
	}
	sort.SliceStable(out, func(a, b int) bool { return out[a]["start_offset"].(int) < out[b]["start_offset"].(int) })
	return text, out, nil
}

// withMentions returns a copy of d with text and a single annotator.
func withMentions(d document, text string, ms []mention) document {
	c := document{}
	for k, v := range d {
		c[k] = v
	}
	list := make([]any, len(ms))
	for i, m := range ms {
		list[i] = m
	}
	c["text"] = text
	c["annotations"] = map[string]any{"annotator1": map[string]any{"entity_mentions": list}}
	return c
}

func readCache(path string) (string, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	return string(b), true
}

// callClaude runs the claude command line without tools, in an empty
// directory, the document on standard input.
func callClaude(model, text string) (string, error) {
	dir, err := os.MkdirTemp("", "translate")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(dir)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "claude", "--print", "--output-format", "json", "--model", model,
		"--tools", "", "--strict-mcp-config", "--no-session-persistence", "--system-prompt", systemPrompt)
	cmd.Dir = dir
	cmd.Stdin = strings.NewReader(text)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("claude: %v: %s", err, strings.TrimSpace(stderr.String()))
	}
	var r struct {
		Result  string `json:"result"`
		IsError bool   `json:"is_error"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &r); err != nil {
		return "", fmt.Errorf("claude output: %v", err)
	}
	if r.IsError {
		return "", fmt.Errorf("claude error: %s", r.Result)
	}
	return r.Result, nil
}

func writeJSON(path string, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}
