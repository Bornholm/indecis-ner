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

// Refine makes masks easier to trust: each mask cut inside a word grows
// to the whole word ("[[é]]crasé" becomes "[[écrasé]]") and leaves out
// the spaces at its edges; empty masks and masks made only of function
// words or punctuation are dropped; masks separated only by spaces,
// commas and function words are joined ("[magasinier] sur le [site]"
// becomes one mask) when join allows it (nil: always), as are masks that
// touch. Masks that overlap are merged whatever join says. Masks come
// back sorted and disjoint.
func Refine(text string, spans []Span, join func(a, b Span) bool) []Span {
	var out []Span
	for _, s := range spans {
		s.Start, s.End = max(s.Start, 0), min(s.End, len(text))
		// An empty mask holds no rune, so cuts no word: it is dropped,
		// wherever it falls.
		if s.Start >= s.End {
			continue
		}
		// An edge inside a rune moves to the rune's bounds: the mask then
		// holds the whole rune, and grows from it as from a cut word.
		for s.Start > 0 && !utf8.RuneStart(text[s.Start]) {
			s.Start--
		}
		for s.End < len(text) && !utf8.RuneStart(text[s.End]) {
			s.End++
		}
		// A mask grows only where it cuts a word: its edge is a letter or
		// a digit, or a sign between two of them. It then grows to the
		// whole word, over such signs too: "des rendez-" and "vous" take
		// "rendez-vous", "/1980" takes "12/03/1980". A mask that starts or
		// ends on a space, or on a comma before a space, does not take the
		// word next to it, which may be another mask's. Where it does, and
		// that word is another mask, the two overlap and are merged
		// whatever join says: a sign inside a word rarely separates two
		// passages, though any sign counts, a full stop without a space
		// after it included.
		first, _ := utf8.DecodeRuneInString(text[s.Start:])
		last, size := utf8.DecodeLastRuneInString(text[:s.End])
		left := isWordRune(first) || joins(text, s.Start)
		right := isWordRune(last) || joins(text, s.End-size)
		for left && s.Start > 0 {
			r, size := utf8.DecodeLastRuneInString(text[:s.Start])
			if !isWordRune(r) && !joins(text, s.Start-size) {
				break
			}
			s.Start -= size
		}
		for right && s.End < len(text) {
			r, size := utf8.DecodeRuneInString(text[s.End:])
			if !isWordRune(r) && !joins(text, s.End) {
				break
			}
			s.End += size
		}
		// Spaces at the edges are left out: two masks that shared the
		// space between them would overlap, and be merged whatever join
		// says. Other signs stay masked, as the % of "80 %".
		for s.Start < s.End {
			r, size := utf8.DecodeRuneInString(text[s.Start:])
			if !unicode.IsSpace(r) {
				break
			}
			s.Start += size
		}
		for s.Start < s.End {
			r, size := utf8.DecodeLastRuneInString(text[s.Start:s.End])
			if !unicode.IsSpace(r) {
				break
			}
			s.End -= size
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

// joins reports whether the rune at i in text is a sign between two
// letters or digits, which joins the parts of a word or a number: the
// hyphen of "rendez-vous", the apostrophe of "l'hôpital", the slashes of
// "12/03/1980", the comma of "1,75".
func joins(text string, i int) bool {
	r, size := utf8.DecodeRuneInString(text[i:])
	if isWordRune(r) || unicode.IsSpace(r) || i == 0 || i+size >= len(text) {
		return false
	}
	before, _ := utf8.DecodeLastRuneInString(text[:i])
	after, _ := utf8.DecodeRuneInString(text[i+size:])
	return isWordRune(before) && isWordRune(after)
}

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
