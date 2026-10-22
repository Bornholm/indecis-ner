package recognizer

import (
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Span is a masked passage, in bytes.
type Span struct{ Start, End int }

// FunctionWords carry no information on their own (articles,
// prepositions, possessives): Refine neither keeps a mask made only of
// them nor leaves them unmasked between two masks.
var FunctionWords = map[string]bool{}

func init() {
	for _, w := range strings.Fields(`le la les l un une des de du d au aux a à en sur dans par pour avec chez sous et ou mais son sa ses mon ma mes ton ta tes notre nos votre vos leur leurs ce cet cette ces qui que qu il elle ils elles on se s y the of a an in on at to and`) {
		FunctionWords[w] = true
	}
}

// Refine makes masks easier to trust: each mask grows to whole words
// ("[[é]]crasé" becomes "[[écrasé]]"), masks made only of function words
// or punctuation are dropped, and masks separated only by spaces, commas
// and function words are joined ("[magasinier] sur le [site]" becomes
// one mask) when join allows it (nil: always), as are masks that touch.
// Masks that overlap are merged whatever join says. Masks come back
// sorted and disjoint.
func Refine(text string, spans []Span, join func(a, b Span) bool) []Span {
	var out []Span
	for _, s := range spans {
		s.Start, s.End = max(s.Start, 0), min(s.End, len(text))
		for s.Start > 0 {
			r, size := utf8.DecodeLastRuneInString(text[:s.Start])
			if !isWordRune(r) {
				break
			}
			s.Start -= size
		}
		for s.End < len(text) {
			r, size := utf8.DecodeRuneInString(text[s.End:])
			if !isWordRune(r) {
				break
			}
			s.End += size
		}
		if s.Start < s.End && informative(text[s.Start:s.End]) {
			out = append(out, s)
		}
	}
	sort.Slice(out, func(a, b int) bool { return out[a].Start < out[b].Start })
	var merged []Span
	for _, s := range out {
		if n := len(merged); n > 0 && (s.Start < merged[n-1].End ||
			filler(text[merged[n-1].End:s.Start]) && (join == nil || join(merged[n-1], s))) {
			merged[n-1].End = max(merged[n-1].End, s.End)
			continue
		}
		merged = append(merged, s)
	}
	return merged
}

func isWordRune(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) }

// words splits a passage into lowercase words.
func words(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool { return !isWordRune(r) })
}

// informative reports whether a passage holds a word that is not a
// function word.
func informative(s string) bool {
	for _, w := range words(s) {
		if !FunctionWords[w] {
			return true
		}
	}
	return false
}

// filler reports whether a gap between two masks holds only spaces,
// commas, apostrophes and at most two function words: what separates the
// parts of one description, not two sentences.
func filler(s string) bool {
	if strings.ContainsAny(s, "\n.!?;:()«»\"") || len(s) > 24 {
		return false
	}
	ws := words(s)
	if len(ws) > 2 {
		return false
	}
	for _, w := range ws {
		if !FunctionWords[w] {
			return false
		}
	}
	return true
}
