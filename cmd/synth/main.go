// Command synth writes synthetic workplace documents labeled by
// construction (see synth and GUIDELINES.md).
//
//	synth generate -n 2000 -out data/synth/work.jsonl
//	synth sample -template bulletin-paie
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"

	"github.com/bornholm/go-anon/pkg/synth/gazetteer"
	"github.com/bornholm/indecis-ner/synth"
	"github.com/bornholm/indecis/dataset"
)

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: synth generate|sample [flags]")
	}
	fs := flag.NewFlagSet(os.Args[1], flag.ExitOnError)
	dir := fs.String("templates", "synth/templates", "template directory")
	n := fs.Int("n", 1000, "documents (generate)")
	seed := fs.Int64("seed", 1, "seed")
	out := fs.String("out", "data/synth/work.jsonl", "output (generate)")
	name := fs.String("template", "", "template to render (sample)")
	only := fs.String("only", "", "comma-separated templates to use (generate; default: all)")
	fs.Parse(os.Args[2:])

	templates, err := synth.LoadTemplates(*dir)
	if err != nil {
		log.Fatal(err)
	}
	bundle, err := gazetteer.LoadSeed("fr", gazetteer.DefaultOptions())
	if err != nil {
		log.Fatal(err)
	}
	lists, err := synth.LoadLists()
	if err != nil {
		log.Fatal(err)
	}
	switch os.Args[1] {
	case "sample":
		for _, t := range templates {
			if t.Name != *name {
				continue
			}
			d, err := synth.Render(t, bundle, lists, *seed)
			if err != nil {
				log.Fatal(err)
			}
			fmt.Println(d.Text)
			fmt.Println("----")
			for _, s := range d.Spans {
				fmt.Printf("%-9s %-6s %-9s %q\n", s.Type, s.Mask, s.Sensitive, d.Text[s.Start:s.End])
			}
			return
		}
		log.Fatalf("no template %q", *name)
	case "generate":
		if *only != "" {
			keep := map[string]bool{}
			for _, t := range strings.Split(*only, ",") {
				keep[t] = true
			}
			var kept []*synth.Template
			for _, t := range templates {
				if keep[t.Name] {
					kept = append(kept, t)
				}
			}
			templates = kept
		}
		docs, err := synth.Generate(templates, bundle, lists, *n, *seed)
		if err != nil {
			log.Fatal(err)
		}
		var ex []dataset.Example
		count := map[string]int{}
		for _, d := range docs {
			ex = append(ex, d.Example())
			count[d.Template]++
		}
		if err := os.MkdirAll(dirOf(*out), 0o755); err != nil {
			log.Fatal(err)
		}
		f, err := os.Create(*out)
		if err != nil {
			log.Fatal(err)
		}
		if err := dataset.WriteJSONL(f, ex); err != nil {
			log.Fatal(err)
		}
		f.Close()
		names := make([]string, 0, len(count))
		for k := range count {
			names = append(names, k)
		}
		sort.Strings(names)
		for _, k := range names {
			log.Printf("%-36s %d", k, count[k])
		}
	}
}

func dirOf(path string) string {
	if i := strings.LastIndex(path, "/"); i >= 0 {
		return path[:i]
	}
	return "."
}
