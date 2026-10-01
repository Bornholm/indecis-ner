// Command ceiling measures what tagging tokens costs before any learning:
// each labeled entity is tagged on the tokens it overlaps, then rebuilt
// from them as an indecis Spans question decodes it (first token without
// its leading space, last token, edge punctuation trimmed). An entity
// whose bounds fall inside a token cannot be found exactly.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/bornholm/indecis/dataset"
	"github.com/bornholm/indecis/tokenizer"
)

func main() {
	home, _ := os.UserHomeDir()
	backbone := flag.String("backbone", home+"/.cache/indecis/models/bekko-embedding-v1-a8m", "backbone directory")
	path := flag.String("test", "data/test_clean.jsonl", "labeled examples")
	show := flag.Int("show", 10, "entities lost to print")
	flag.Parse()
	tok, err := tokenizer.Load(*backbone + "/tokenizer.json")
	if err != nil {
		log.Fatal(err)
	}
	ex, err := dataset.ReadFile(*path)
	if err != nil {
		log.Fatal(err)
	}
	total, exact, shown := 0, 0, 0
	for _, e := range ex {
		_, offs := tok.AppendOffsets(nil, nil, e.Text)
		list, _ := e.Labels["entities"].([]any)
		for _, item := range list {
			m := item.(map[string]any)
			gs, ge := int(m["start"].(float64)), int(m["end"].(float64))
			s, end := -1, -1
			for _, o := range offs {
				cs, ce := core(e.Text, o)
				if cs < ge && gs < ce {
					if s < 0 {
						s = cs
					}
					end = o.End
				}
			}
			total++
			if s >= 0 {
				s, end = trim(e.Text, s, end)
			}
			if s == gs && end == ge {
				exact++
				continue
			}
			if shown < *show {
				shown++
				got := ""
				if s >= 0 {
					got = e.Text[s:end]
				}
				fmt.Printf("  %q -> %q\n", e.Text[gs:ge], got)
			}
		}
	}
	fmt.Printf("%s: %d/%d entities rebuilt exactly (%.2f%%)\n", *path, exact, total, 100*float64(exact)/float64(total))
}

func core(text string, o tokenizer.Offset) (int, int) {
	s := o.Start
	for s < o.End {
		r, size := utf8.DecodeRuneInString(text[s:o.End])
		if !unicode.IsSpace(r) {
			break
		}
		s += size
	}
	if s == o.End {
		return o.Start, o.End
	}
	return s, o.End
}

const (
	openPunct  = "([{«‹“‘\"'¿¡"
	closePunct = ",;:!?)]}»›”’\"'"
)

func trim(text string, start, end int) (int, int) {
	s, e := start, end
	for s < e {
		r, size := utf8.DecodeRuneInString(text[s:e])
		if !strings.ContainsRune(openPunct, r) && !unicode.IsSpace(r) {
			break
		}
		s += size
	}
	for s < e {
		r, size := utf8.DecodeLastRuneInString(text[s:e])
		if !unicode.IsSpace(r) && (!strings.ContainsRune(closePunct, r) || paired(text[s:e-size], r)) {
			break
		}
		e -= size
	}
	if s == e {
		return start, end
	}
	return s, e
}

func paired(span string, closing rune) bool {
	open, ok := map[rune]rune{')': '(', ']': '[', '}': '{', '»': '«', '›': '‹', '”': '“'}[closing]
	return ok && strings.Count(span, string(open)) > strings.Count(span, string(closing))
}
