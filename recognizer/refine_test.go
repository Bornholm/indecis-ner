package recognizer

import (
	"slices"
	"strings"
	"testing"

	"github.com/bornholm/indecis"
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

// A sensitive passage keeps its category in its mask: the masks around
// it are not joined to it, and a mask that holds it, however long, takes
// it. The model's spans for a sentence of the demonstration document of
// indecis-caviardage, where the joined mask once came out as a place.
func TestMasksSensitive(t *testing.T) {
	text := "Elle fréquente avec ses enfants à la synagogue de Chalon. Son fils aîné est suivi au CMP pour des troubles de l'attention."
	at := func(s string, typ string) indecis.Span {
		i := strings.Index(text, s)
		if i < 0 {
			t.Fatalf("%q not in text", s)
		}
		return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ}
	}
	found := []indecis.Span{
		at("fréquente avec", "QUASI"),
		at("ses enfants à la synagogue de Chalon", "QUASI"),
		at("Son fils aîné est suivi au CMP pour des troubles de l'attention", "QUASI"),
	}
	sensitive := []indecis.Span{
		at("ses", "BELIEF"), // a function word alone gives no category
		at("la synagogue de Chalon", "BELIEF"),
		at("troubles de l'attention", "HEALTH"),
	}
	found = append(found, sensitive...)
	typed := []indecis.Span{at("ses enfants", "MISC"), at("la synagogue de Chalon", "LOC"), at("Son fils aîné", "DEM")}
	var got []string
	for _, m := range Masks(text, found, typed, sensitive) {
		got = append(got, text[m.Start:m.End]+" "+m.Sensitive)
	}
	want := []string{
		"fréquente avec ",
		"ses enfants à la synagogue de Chalon BELIEF",
		"Son fils aîné est suivi au CMP pour des troubles de l'attention HEALTH",
	}
	if !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}
