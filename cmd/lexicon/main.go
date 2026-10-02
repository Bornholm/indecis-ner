// Command lexicon records the words of the passages of training sets, by
// category of a question, and writes the most frequent ones as a list of
// words a test set must avoid (prose -forbid).
//
//	lexicon -question sensitive -out data/lexicon data/synth2/train.jsonl data/prose/*train*.jsonl
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/bornholm/indecis-ner/lexicon"
	"github.com/bornholm/indecis/dataset"
)

func main() {
	question := flag.String("question", "sensitive", "label holding the passages")
	out := flag.String("out", "data/lexicon", "output directory: lexicon.json, words.txt")
	top := flag.Int("top", 60, "words per category in words.txt")
	only := flag.String("categories", "", "categories listed in words.txt, comma-separated (default: all)")
	skip := flag.String("skip", "", "file of words never written to words.txt, too common to forbid")
	flag.Parse()
	lex := lexicon.Lexicon{}
	surface := map[string]map[string]int{} // stem -> word forms, for the prompt
	for _, path := range flag.Args() {
		ex, err := dataset.ReadFile(path)
		if err != nil {
			log.Fatal(err)
		}
		for _, e := range ex {
			list, _ := e.Labels[*question].([]any)
			for _, item := range list {
				m := item.(map[string]any)
				s, en := int(m["start"].(float64)), int(m["end"].(float64))
				passage := e.Text[s:en]
				lex.Add(m["type"].(string), passage)
				for _, w := range lexicon.Words(passage) {
					st := lexicon.Stem(w)
					if surface[st] == nil {
						surface[st] = map[string]int{}
					}
					surface[st][w]++
				}
			}
		}
		log.Printf("%s: %d examples", path, len(ex))
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}
	if err := lex.Save(filepath.Join(*out, "lexicon.json")); err != nil {
		log.Fatal(err)
	}
	var cats []string
	for c := range lex {
		cats = append(cats, c)
	}
	sort.Strings(cats)
	seen := map[string]bool{}
	if *skip != "" {
		b, err := os.ReadFile(*skip)
		if err != nil {
			log.Fatal(err)
		}
		for _, w := range strings.Fields(string(b)) {
			seen[w] = true
		}
	}
	keep := map[string]bool{}
	for _, c := range strings.Split(*only, ",") {
		if c != "" {
			keep[c] = true
		}
	}
	var words []string
	for _, c := range cats {
		if len(keep) > 0 && !keep[c] {
			continue
		}
		var forms []string
		for _, st := range lex.Top(c, *top) {
			best, n := "", 0
			for w, k := range surface[st] {
				if k > n || k == n && w < best {
					best, n = w, k
				}
			}
			if !seen[best] {
				seen[best] = true
				words = append(words, best)
				forms = append(forms, best)
			}
		}
		fmt.Printf("%-10s %4d stems; top: %s\n", c, len(lex[c]), strings.Join(forms[:min(15, len(forms))], ", "))
	}
	if err := os.WriteFile(filepath.Join(*out, "words.txt"), []byte(strings.Join(words, "\n")+"\n"), 0o644); err != nil {
		log.Fatal(err)
	}
	log.Printf("%d forbidden words", len(words))
}
