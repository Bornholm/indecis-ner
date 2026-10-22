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

// A mask covered by two special categories carries the one it overlaps
// most.
func TestMasksTwoCategories(t *testing.T) {
	text := "Il jeûne pendant le ramadan malgré son diabète."
	at := func(s, typ string) indecis.Span {
		i := strings.Index(text, s)
		return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ}
	}
	whole := at("jeûne pendant le ramadan malgré son diabète", "QUASI")
	sensitive := []indecis.Span{at("diabète", "HEALTH"), at("jeûne pendant le ramadan", "BELIEF")}
	got := Masks(text, []indecis.Span{whole}, nil, sensitive)
	if len(got) != 1 || got[0].Sensitive != "BELIEF" {
		t.Errorf("got %+v, want one mask flagged BELIEF", got)
	}
}

// Offsets past the text, which Raw keeps as the model gives them, give a
// category without slicing out of the text.
func TestCategoryOutOfText(t *testing.T) {
	text := "suivi pour un diabète"
	beyond := indecis.Span{Start: 14, End: len(text) + 8, Type: "HEALTH"}
	if got := category(text, Span{0, len(text) + 4}, []indecis.Span{beyond}); got != "HEALTH" {
		t.Errorf("got %q, want HEALTH", got)
	}
}

// A sensitive passage under MinConfidence neither flags a mask nor keeps
// it from being joined to its neighbour.
func TestMinConfidenceSensitive(t *testing.T) {
	text := "Elle est infirmière de nuit en pneumologie."
	at := func(s, typ string, conf float64) indecis.Span {
		i := strings.Index(text, s)
		return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ, Confidence: conf}
	}
	masks := []indecis.Span{at("infirmière de nuit", "QUASI", 0.9), at("pneumologie", "QUASI", 0.9)}
	for _, c := range []struct {
		conf float64
		want []string
	}{
		{0.2, []string{"infirmière de nuit en pneumologie "}},
		{0.8, []string{"infirmière de nuit ", "pneumologie HEALTH"}},
	} {
		sensitive := []indecis.Span{at("pneumologie", "HEALTH", c.conf)}
		r := &Recognizer{MinConfidence: 0.5, known: map[string][]decided{
			text: {{masks: append(slices.Clone(masks), sensitive...), sensitive: sensitive}},
		}}
		var got []string
		r.OnFound = func(_ string, found []Found) {
			for _, f := range found {
				got = append(got, f.Text+" "+f.Sensitive)
			}
		}
		if _, err := r.Recognize(text); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, c.want) {
			t.Errorf("sensitive confidence %g: got %q, want %q", c.conf, got, c.want)
		}
	}
}

// A sensitive passage keeps its category in its mask: the masks around
// it are not joined to it, and a mask that holds it, however long, takes
// it. Masks of one category are still joined, names still are not. The model's spans for a sentence of the demonstration document of
// indecis-caviardage, where the joined mask once came out as a place.
func TestMasksSensitive(t *testing.T) {
	text := "Elle fréquente avec ses enfants à la synagogue de Chalon. Son fils aîné est suivi au CMP pour des troubles de l'attention. " +
		"Elle a une lombalgie chronique et un asthme. Mme Roux, infirmière, la remplace."
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
		at("lombalgie chronique", "QUASI"),
		at("asthme", "QUASI"),
		at("Mme Roux", "DIRECT"),
		at("infirmière", "QUASI"),
	}
	sensitive := []indecis.Span{
		at("ses", "BELIEF"), // a function word alone gives no category
		at("la synagogue de Chalon", "BELIEF"),
		at("troubles de l'attention", "HEALTH"),
		at("lombalgie chronique", "HEALTH"),
		at("asthme", "HEALTH"),
	}
	found = append(found, sensitive...)
	typed := []indecis.Span{at("ses enfants", "MISC"), at("la synagogue de Chalon", "LOC"), at("Son fils aîné", "DEM"),
		at("Mme Roux", "PERSON"), at("infirmière", "DEM")}
	var got []string
	for _, m := range Masks(text, found, typed, sensitive) {
		got = append(got, text[m.Start:m.End]+" "+m.Sensitive)
	}
	want := []string{
		"fréquente avec ",
		"ses enfants à la synagogue de Chalon BELIEF",
		"Son fils aîné est suivi au CMP pour des troubles de l'attention HEALTH",
		// Two masks of the same category are joined over function words.
		"lombalgie chronique et un asthme HEALTH",
		// A name is not, though neither is sensitive.
		"Mme Roux ",
		"infirmière ",
	}
	if !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}
