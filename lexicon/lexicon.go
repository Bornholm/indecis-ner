// Package lexicon records the words of the passages of a training set,
// by category, so that a test set can be checked for passages whose words
// the model never saw in that category: a score on those measures what
// the model understood, not which words it memorized.
//
// Words are compared by stem: lowercase, without accents, the first six
// letters ("musulmane" and "musulman" share "musulm"). Stop words and
// words shorter than four letters are left out.
package lexicon

import (
	"encoding/json"
	"os"
	"sort"
	"strings"
	"unicode"

	"golang.org/x/text/runes"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

// Lexicon maps a category to the stems seen in its passages, with counts.
type Lexicon map[string]map[string]int

var stop = map[string]bool{}

func init() {
	for _, w := range strings.Fields(`avec dans pour sans sous chez entre vers depuis pendant contre leur leurs elle elles nous vous ils sont était etait être etre avoir avait fait faire cette ces son sa ses une des les aux par sur qui que quoi dont mais donc car comme plus moins tres très tout tous toute toutes autre autres même meme lors alors aussi encore`) {
		stop[fold(w)] = true
	}
}

func fold(s string) string {
	t := transform.Chain(norm.NFD, runes.Remove(runes.In(unicode.Mn)), norm.NFC)
	out, _, _ := transform.String(t, strings.ToLower(s))
	return out
}

// Words returns the content words of a passage, folded.
func Words(s string) []string {
	var out []string
	for _, w := range strings.FieldsFunc(fold(s), func(r rune) bool { return !unicode.IsLetter(r) }) {
		if len([]rune(w)) >= 4 && !stop[w] {
			out = append(out, w)
		}
	}
	return out
}

// Stem is the first six letters of a folded word.
func Stem(w string) string {
	r := []rune(w)
	if len(r) > 6 {
		r = r[:6]
	}
	return string(r)
}

// Add records a passage of a category.
func (l Lexicon) Add(category, passage string) {
	if l[category] == nil {
		l[category] = map[string]int{}
	}
	for _, w := range Words(passage) {
		l[category][Stem(w)]++
	}
}

// Novel reports whether none of the words of a passage was seen in its
// category.
func (l Lexicon) Novel(category, passage string) bool {
	for _, w := range Words(passage) {
		if l[category][Stem(w)] > 0 {
			return false
		}
	}
	return true
}

// Top returns the most frequent stems of a category.
func (l Lexicon) Top(category string, n int) []string {
	var stems []string
	for s := range l[category] {
		stems = append(stems, s)
	}
	sort.Slice(stems, func(a, b int) bool {
		ca, cb := l[category][stems[a]], l[category][stems[b]]
		return ca > cb || ca == cb && stems[a] < stems[b]
	})
	if len(stems) > n {
		stems = stems[:n]
	}
	return stems
}

// Load reads a lexicon written by Save.
func Load(path string) (Lexicon, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var l Lexicon
	return l, json.Unmarshal(b, &l)
}

// Save writes the lexicon as JSON.
func (l Lexicon) Save(path string) error {
	b, err := json.MarshalIndent(l, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}
