// Package recognizer puts an indecis model behind go-anon's recognizer
// interface (anonymizer.Recognizer), in place of the CRF: the
// anonymizer, its mapping and its strict verification stay go-anon's.
//
//	m, _ := indecis.Load("runs/a8m-full", indecis.WithInt8())
//	r := recognizer.New(m, "entities",
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

// Recognizer finds the entities of a text with a Spans question of an
// indecis model, then applies go-anon's filters (identifiers by regular
// expression, secrets, confidence threshold), as its own recognizer does
// after the CRF.
type Recognizer struct {
	model    *indecis.Model
	question string
	filters  []ner.EntityFilter
	// MinConfidence drops the passages whose confidence is lower.
	MinConfidence float64
}

// New wraps model, whose question is a Spans question with go-anon's
// types (PER, LOC, ORG, MISC).
func New(model *indecis.Model, question string, filters ...ner.EntityFilter) *Recognizer {
	return &Recognizer{model: model, question: question, filters: filters}
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
	a, ok := d[0][r.question]
	if !ok || a.Kind != indecis.Spans {
		return nil, fmt.Errorf("recognizer: the model has no spans question %q", r.question)
	}
	out := make([]ner.Entity, 0, len(a.Spans))
	for _, s := range a.Spans {
		if s.Confidence < r.MinConfidence {
			continue
		}
		out = append(out, ner.Entity{Text: s.Text, Type: ner.EntityType(s.Type), Start: s.Start, End: s.End, Confidence: s.Confidence})
	}
	for _, f := range r.filters {
		out = f(text, out)
	}
	return out, nil
}
