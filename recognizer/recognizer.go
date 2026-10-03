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
}

// Found is an entity returned by Recognize, with the special category
// (GDPR article 9 or 10) of the sensitive passage it covers, if any.
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

// Recognize implements anonymizer.Recognizer.
func (r *Recognizer) Recognize(text string) ([]ner.Entity, error) {
	return r.RecognizeContext(context.Background(), text)
}

// RecognizeContext is Recognize with a context.
func (r *Recognizer) RecognizeContext(ctx context.Context, text string) ([]ner.Entity, error) {
	if text == "" {
		return []ner.Entity{}, nil
	}
	d, err := r.model.Decide(ctx, text)
	if err != nil {
		return nil, err
	}
	var found, sensitive []indecis.Span
	for _, q := range r.questions {
		a, ok := d[0][q]
		if !ok || a.Kind != indecis.Spans {
			return nil, fmt.Errorf("recognizer: the model has no spans question %q", q)
		}
		for _, s := range a.Spans {
			if s.Confidence >= r.MinConfidence {
				found = append(found, s)
				if q == "sensitive" {
					sensitive = append(sensitive, s)
				}
			}
		}
	}
	var out []ner.Entity
	if r.Raw {
		for _, s := range found {
			out = append(out, ner.Entity{Text: s.Text, Type: ner.EntityType(s.Type), Start: s.Start, End: s.End, Confidence: s.Confidence})
		}
	} else {
		for _, m := range Masks(text, found, d[0]["entities"].Spans) {
			out = append(out, ner.Entity{Text: text[m.Start:m.End], Type: m.Type, Start: m.Start, End: m.End, Confidence: m.Confidence})
		}
	}
	for _, f := range r.filters {
		out = f(text, out)
	}
	if r.OnFound != nil {
		list := make([]Found, len(out))
		for i, e := range out {
			list[i] = Found{Entity: e, Sensitive: category(Span{e.Start, e.End}, sensitive)}
		}
		r.OnFound(text, list)
	}
	return out, nil
}

// category returns the type of the sensitive passage that overlaps s the
// most, "" if none.
func category(s Span, sensitive []indecis.Span) string {
	cat, best := "", 0
	for _, t := range sensitive {
		if o := min(s.End, t.End) - max(s.Start, t.Start); o > best {
			cat, best = t.Type, o
		}
	}
	return cat
}

// Mask is a refined mask and its type.
type Mask struct {
	Span
	Type       ner.EntityType
	Confidence float64
}

// Masks refines the passages found by the mask questions (Refine) and
// types each mask from the typed passages of the entities question.
// Masks of a name or an identifier are not joined to their neighbours:
// go-anon gives each name its own consistent placeholder.
func Masks(text string, found, typed []indecis.Span) []Mask {
	spans := make([]Span, len(found))
	for i, s := range found {
		spans[i] = Span{s.Start, s.End}
	}
	own := func(s Span) bool {
		t, _ := typeOf(s, typed, nil)
		return t == "PERSON" || t == "CODE"
	}
	var out []Mask
	for _, s := range Refine(text, spans, func(a, b Span) bool { return !own(a) && !own(b) }) {
		typ, conf := typeOf(s, typed, found)
		out = append(out, Mask{s, typ, conf})
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
