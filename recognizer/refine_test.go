package recognizer

import (
	"slices"
	"testing"
)

func TestRefine(t *testing.T) {
	text := "Il a été écrasé par une charge. Magasinier sur le site de Stains.\nMme Roux"
	// "charge. Magasinier": a sentence ends in between, no join.
	at := func(s string) Span {
		for i := range text {
			if len(text)-i >= len(s) && text[i:i+len(s)] == s {
				return Span{i, i + len(s)}
			}
		}
		t.Fatalf("%q not in text", s)
		return Span{}
	}
	cut := func(s string, from, to int) Span { sp := at(s); return Span{sp.Start + from, sp.Start + to} }
	got := Refine(text, []Span{
		cut("écrasé", 0, 2),                        // "é": grows to the word
		at("par"),                                  // function word only: dropped
		at("charge"),                               // joined to "écrasé" over "par une"
		at("Magasinier"), at("site"), at("Stains"), // joined over "sur le", "de"
		at("Mme Roux"), // not joined over the line break
	}, nil)
	want := []Span{{at("écrasé").Start, at("charge").End}, {at("Magasinier").Start, at("Stains").End}, at("Mme Roux")}
	if !slices.Equal(got, want) {
		for _, s := range got {
			t.Logf("%q", text[s.Start:s.End])
		}
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestSteadyPlaceholders(t *testing.T) {
	in := "Bonjour ⟦PERSON_1_c471b2⟧, voir ⟦CODE_12_0a9f⟧ et ⟦x⟧."
	got := steadyPlaceholders(in)
	want := "Bonjour ⟦PERSON_1_xxxxxx⟧, voir ⟦CODE_12_xxxx⟧ et ⟦x⟧."
	if got != want || len(got) != len(in) {
		t.Fatalf("got %q", got)
	}
}
