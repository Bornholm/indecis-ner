// Package recognizer puts an indecis model behind go-anon's recognizer
// interface (anonymizer.Recognizer), in place of the CRF: the
// anonymizer, its mapping and its strict verification stay go-anon's.
//
// Two modes. A model trained on named entities (WikiNER, the "entities"
// question with PER, LOC, ORG, MISC) replaces the CRF as is. A model
// trained to mask personal data (the "mask" and "sensitive" questions,
// GUIDELINES.md) masks the union of both, refined (see Refine), each mask
// typed from the "entities" question: what matters there is that
// nothing leaks, and the decoding bias sets how far.
//
//	m, _ := indecis.Load("runs/work-ind", indecis.WithInt8(),
//		indecis.WithSpanBias("mask", 5), indecis.WithSpanBias("sensitive", 5))
//	r := recognizer.New(m, []string{"mask", "sensitive"},
//		ner.RegexEntityFilter(ner.BuiltinRegexPatterns),
//		ner.RegexEntityFilter(ner.SecretPatterns()))
//	anon := anonymizer.New(r, anonymizer.Config{Strategy: anonymizer.TagReplace})
package recognizer

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/bornholm/go-anon/pkg/ner"
	"github.com/bornholm/indecis"
)

// Recognizer finds the passages of a text with Spans questions of an
// indecis model, then applies go-anon's filters (identifiers by regular
// expression, secrets, confidence threshold), as its own recognizer does
// after the CRF.
type Recognizer struct {
	model     *indecis.Model
	questions []string
	filters   []ner.EntityFilter
	// MinConfidence drops the passages whose confidence is lower.
	MinConfidence float64
	// Raw keeps the passages as the model decodes them, typed by their
	// own question, instead of refining their union.
	Raw bool
	// OnFound, if set, receives each text given to Recognize and what was
	// found in it, go-anon's filters included: to list what a document
	// had masked.
	OnFound func(text string, found []Found)

	// known holds the passages found by Document in each paragraph.
	known map[string][]decided
}

// decided is a paragraph's share of a whole-document decision.
type decided struct {
	masks, sensitive, typed []indecis.Span
}

// Found is an entity returned by Recognize, with the special category
// (GDPR article 9 or 10) of the sensitive passage it covers, if any: one
// category only, that of the passage it overlaps most, when it covers
// several.
type Found struct {
	ner.Entity
	Sensitive string
}

// New wraps model. With one question whose types are go-anon's (PER,
// LOC, ORG, MISC), set Raw. With mask questions ("mask", "sensitive"),
// their passages are joined and refined, and typed from the model's
// "entities" question when it has one.
func New(model *indecis.Model, questions []string, filters ...ner.EntityFilter) *Recognizer {
	return &Recognizer{model: model, questions: questions, filters: filters}
}

// Document reads the paragraphs of a document as one text, joined by
// line breaks, the way the model was trained, and keeps what it finds in
// each. go-anon then gives Recognize one paragraph at a time: a known
// paragraph gets the passages found with the whole document as context,
// so a heading or a short line is not judged alone. Other texts
// (verification passes) are decided on their own.
func (r *Recognizer) Document(ctx context.Context, paragraphs []string) error {
	text := strings.Join(paragraphs, "\n")
	d, err := r.model.Decide(ctx, text)
	if err != nil {
		return err
	}
	r.known = map[string][]decided{}
	start := 0
	for _, p := range paragraphs {
		end := start + len(p)
		var dec decided
		shift := func(spans []indecis.Span) []indecis.Span {
			var out []indecis.Span
			for _, s := range spans {
				s.Start, s.End = max(s.Start, start), min(s.End, end)
				if s.Start >= s.End {
					continue
				}
				s.Start, s.End = s.Start-start, s.End-start
				s.Text = p[s.Start:s.End]
				out = append(out, s)
			}
			return out
		}
		for _, q := range r.questions {
			found := shift(d[0][q].Spans)
			dec.masks = append(dec.masks, found...)
			if q == "sensitive" {
				dec.sensitive = append(dec.sensitive, found...)
			}
		}
		dec.typed = shift(d[0]["entities"].Spans)
		if _, dup := r.known[p]; !dup {
			r.known[p] = []decided{dec}
		}
		start = end + 1
	}
	return nil
}

// Recognize implements anonymizer.Recognizer.
func (r *Recognizer) Recognize(text string) ([]ner.Entity, error) {
	return r.RecognizeContext(context.Background(), text)
}

// RecognizeContext is Recognize with a context.
func (r *Recognizer) RecognizeContext(ctx context.Context, text string) ([]ner.Entity, error) {
	if text == "" {
		return []ner.Entity{}, nil
	}
	var found, sensitive, typed []indecis.Span
	if k, ok := r.known[text]; ok {
		found, sensitive, typed = k[0].masks, k[0].sensitive, k[0].typed
	} else {
		d, err := r.model.Decide(ctx, steadyPlaceholders(text))
		if err != nil {
			return nil, err
		}
		for _, q := range r.questions {
			a, ok := d[0][q]
			if !ok || a.Kind != indecis.Spans {
				return nil, fmt.Errorf("recognizer: the model has no spans question %q", q)
			}
			found = append(found, a.Spans...)
			if q == "sensitive" {
				sensitive = append(sensitive, a.Spans...)
			}
		}
		typed = d[0]["entities"].Spans
	}
	// The sensitive passages too: below the threshold, they neither block
	// a join nor flag a mask.
	confident := func(spans []indecis.Span) []indecis.Span {
		kept := spans[:0:0]
		for _, s := range spans {
			if s.Confidence >= r.MinConfidence {
				kept = append(kept, s)
			}
		}
		return kept
	}
	found, sensitive = confident(found), confident(sensitive)
	var out []ner.Entity
	if r.Raw {
		for _, s := range found {
			out = append(out, ner.Entity{Text: s.Text, Type: ner.EntityType(s.Type), Start: s.Start, End: s.End, Confidence: s.Confidence})
		}
	} else {
		for _, m := range Masks(text, found, typed, sensitive) {
			out = append(out, ner.Entity{Text: text[m.Start:m.End], Type: m.Type, Start: m.Start, End: m.End, Confidence: m.Confidence})
		}
	}
	for _, f := range r.filters {
		out = f(text, out)
	}
	if r.OnFound != nil {
		list := make([]Found, len(out))
		for i, e := range out {
			list[i] = Found{Entity: e, Sensitive: category(text, Span{e.Start, e.End}, sensitive)}
		}
		r.OnFound(text, list)
	}
	return out, nil
}

// placeholderNonce matches the random suffix of go-anon's placeholders,
// ⟦TYPE_N_nonce⟧.
var placeholderNonce = regexp.MustCompile(`(⟦[A-Z][A-Z0-9_]*?_\d+_)([0-9a-f]+)(⟧)`)

// steadyPlaceholders replaces the random suffix of go-anon's placeholders
// with as many fixed letters. Verification passes read text full of
// placeholders, and a suffix that changes at every run made what the
// model found around them change too. The length, hence every offset,
// stays the same.
func steadyPlaceholders(text string) string {
	if !strings.Contains(text, "⟦") {
		return text
	}
	return placeholderNonce.ReplaceAllStringFunc(text, func(m string) string {
		p := placeholderNonce.FindStringSubmatch(m)
		return p[1] + strings.Repeat("x", len(p[2])) + p[3]
	})
}

// category returns the type of the sensitive passage that overlaps s the
// most, however little of s it covers, as long as the overlap holds a
// word other than a function word: a mask that holds a special category
// of data is sensitive as a whole. An overlap of function words alone
// gives no category, so it does not keep Masks from joining s.
func category(text string, s Span, sensitive []indecis.Span) string {
	cat, best := "", 0
	for _, t := range sensitive {
		// The model's offsets, which Raw keeps as they are, are not
		// trusted to fall within text.
		start, end := max(s.Start, t.Start, 0), min(s.End, t.End, len(text))
		if o := end - start; o > best && informative(text[start:end]) {
			cat, best = t.Type, o
		}
	}
	return cat
}

// Mask is a refined mask, its type and its special category, if any.
type Mask struct {
	Span
	Type       ner.EntityType
	Confidence float64
	Sensitive  string
}

// Masks refines the passages found by the mask questions (Refine), types
// each mask from the typed passages of the entities question and gives
// it the special category of the sensitive passages. Masks of a name or
// an identifier are not joined to their neighbours over function words:
// go-anon gives each name its own consistent placeholder. Nor are masks of
// different special categories, or a sensitive mask and one that is not.
// Masks that overlap are merged whatever they hold: a mask that holds a
// sensitive passage stays flagged, but with one category only, that of
// the passage it overlaps most. That category is the merged mask's own:
// joining two masks of one category may give a mask whose largest
// overlap is a third passage, of another category, which it then
// carries. It is never left without one.
func Masks(text string, found, typed, sensitive []indecis.Span) []Mask {
	spans := make([]Span, len(found))
	for i, s := range found {
		spans[i] = Span{s.Start, s.End}
	}
	own := func(s Span) bool {
		t, _ := typeOf(s, typed, nil)
		return t == "PERSON" || t == "CODE"
	}
	join := func(a, b Span) bool {
		return !own(a) && !own(b) && category(text, a, sensitive) == category(text, b, sensitive)
	}
	var out []Mask
	for _, s := range Refine(text, spans, join) {
		typ, conf := typeOf(s, typed, found)
		out = append(out, Mask{s, typ, conf, category(text, s, sensitive)})
	}
	return out
}

// typeOf names a mask after the typed passage it overlaps most, MISC if
// none, with the best confidence of the passages it was built from.
func typeOf(s Span, typed, found []indecis.Span) (ner.EntityType, float64) {
	typ, best := "MISC", 0
	for _, t := range typed {
		if o := min(s.End, t.End) - max(s.Start, t.Start); o > best {
			typ, best = t.Type, o
		}
	}
	var conf float64
	for _, f := range found {
		if f.Start < s.End && s.Start < f.End {
			conf = max(conf, f.Confidence)
		}
	}
	return ner.EntityType(typ), conf
}
