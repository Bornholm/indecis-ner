package recognizer

import (
	"slices"
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
// the spaces, the punctuation of sentences and the unpaired brackets at
// its edges (trim); empty masks and masks made only of function words or
// punctuation are dropped; masks separated only by spaces, commas and
// function words are joined ("[magasinier] sur le [site]" becomes one
// mask) when join allows it (nil: always), as are masks that touch. A
// comma left out of an edge thus still joins two masks, a full stop, a
// semicolon or a colon does not. Masks that overlap are merged whatever
// join says. Masks come back sorted and disjoint.
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
		s = trim(text, s)
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

// trim leaves the spaces and the punctuation of sentences out of a
// mask's edges: two masks that shared the space between them would
// overlap, and be merged whatever join says, and the full stop after a
// number is not part of it. Other signs stay masked, as the % of "80 %".
// So is a bracket that is not closed, or opened, inside the mask:
// "(1 245,60 €)" masked as "1 245,60 €)" leaves "(" readable, and then
// ")"; "(annexe 2) (" leaves the last "(" readable.
func trim(text string, s Span) Span {
	// Leaving a bracket out may leave its partner unpaired, at the other
	// edge: trim until nothing moves.
	for {
		before := s
		for s.Start < s.End {
			r, size := utf8.DecodeRuneInString(text[s.Start:])
			if !edge(r) && !unpaired(text[s.Start:s.End], r, true) {
				break
			}
			s.Start += size
		}
		for s.Start < s.End {
			r, size := utf8.DecodeLastRuneInString(text[s.Start:s.End])
			if !edge(r) && !unpaired(text[s.Start:s.End], r, false) {
				break
			}
			s.End -= size
		}
		if s == before {
			return s
		}
	}
}

func isWordRune(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) }

// edge reports whether Refine leaves r out of a mask's edges: spaces and
// the punctuation of sentences.
func edge(r rune) bool { return unicode.IsSpace(r) || strings.ContainsRune(".,;:!?…", r) }

// closing maps each opening bracket to its partner, opening each
// closing one to its partner.
var (
	closing = map[rune]rune{'(': ')', '[': ']', '«': '»'}
	opening = map[rune]rune{')': '(', ']': '[', '»': '«'}
)

// unpaired reports whether r, a bracket at the start of mask (or at its
// end), is not closed (opened) inside it: counted by depth, so the ")" of
// another pair does not close the "(" of "(annexe 2) (".
func unpaired(mask string, r rune, start bool) bool {
	if partner, ok := closing[r]; ok {
		return !start || !closed([]rune(mask), r, partner)
	}
	if partner, ok := opening[r]; ok {
		rs := []rune(mask)
		slices.Reverse(rs)
		return start || !closed(rs, r, partner)
	}
	return false
}

// closed reports whether the bracket r that starts rs finds its partner
// in them.
func closed(rs []rune, r, partner rune) bool {
	depth := 0
	for _, x := range rs {
		switch x {
		case r:
			depth++
		case partner:
			if depth--; depth == 0 {
				return true
			}
		}
	}
	return false
}

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
