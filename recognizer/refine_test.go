package recognizer

import (
	"slices"
	"strings"
	"testing"

	"github.com/bornholm/go-anon/pkg/ner"
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

// A mask that starts or ends on a space or a comma does not grow over
// the word next to it (#2): only a mask cut inside a word grows. Spaces
// at the edges are left out.
func TestRefineEdges(t *testing.T) {
	text := "Elle consulte pour son diabète, avant la prière du vendredi."
	at := func(s string) Span {
		i := strings.Index(text, s)
		if i < 0 {
			t.Fatalf("%q not in text", s)
		}
		return Span{i, i + len(s)}
	}
	for _, c := range []struct {
		in   Span
		want string
	}{
		{at(" avant la prière"), "avant la prière"},
		{at("diabète, "), "diabète,"},
		{at(", avant"), ", avant"},
		{at("abète, av"), "diabète, avant"},
		// A cut inside a word that holds a two-byte rune, "ièr", and a cut
		// inside that rune, "è" being bytes 3 and 4 of "prière".
		{Span{at("prière").Start + 2, at("prière").End - 1}, "prière"},
		{Span{at("prière").Start + 4, at("prière").Start + 4 + 1}, "prière"},
		{Span{at("prière").Start, at("prière").Start + 4}, "prière"},
		{at("son diabète, avant"), "son diabète, avant"},
	} {
		got := Refine(text, []Span{c.in}, nil)
		if len(got) != 1 || text[got[0].Start:got[0].End] != c.want {
			t.Errorf("%q: got %v, want %q", text[c.in.Start:c.in.End], got, c.want)
		}
	}
	// An empty mask, even inside a word, cuts none and is dropped.
	if got := Refine(text, []Span{{at("prière").Start + 2, at("prière").Start + 2}}, nil); len(got) != 0 {
		t.Errorf("empty mask: got %v, want none", got)
	}
	// A hyphen or an apostrophe at the edge cuts a word.
	text = "Des rendez-vous à l'hôpital."
	for _, c := range []struct {
		in   Span
		want string
	}{
		{at("Des rendez-"), "Des rendez-vous"},
		{at("-vous à"), "rendez-vous à"},
		{at("'hôpital"), "l'hôpital"},
	} {
		got := Refine(text, []Span{c.in}, nil)
		if len(got) != 1 || text[got[0].Start:got[0].End] != c.want {
			t.Errorf("%q: got %v, want %q", text[c.in.Start:c.in.End], got, c.want)
		}
	}
}

// The example of #2: a mask that starts on a space no longer takes the
// sensitive word before it, which keeps its own category.
func TestMasksLeadingSpace(t *testing.T) {
	text := "Elle consulte pour son diabète avant la prière du vendredi."
	at := func(s, typ string) indecis.Span {
		i := strings.Index(text, s)
		return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ}
	}
	diabetes := at("diabète", "HEALTH")
	found := []indecis.Span{diabetes, at(" avant la prière du vendredi", "QUASI")}
	sensitive := []indecis.Span{diabetes, at("prière du vendredi", "BELIEF")}
	var got []string
	for _, m := range Masks(text, found, nil, sensitive) {
		got = append(got, text[m.Start:m.End]+" "+m.Sensitive)
	}
	want := []string{"diabète HEALTH", "avant la prière du vendredi BELIEF"}
	if !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
	// Two masks that share the space between them: without it, they no
	// longer overlap, and join keeps them apart.
	shared := diabetes
	shared.End++
	got = nil
	for _, m := range Masks(text, []indecis.Span{shared, found[1]}, nil, []indecis.Span{shared, sensitive[1]}) {
		got = append(got, text[m.Start:m.End]+" "+m.Sensitive)
	}
	if !slices.Equal(got, want) {
		t.Errorf("shared space: got %q, want %q", got, want)
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

// Masks that overlap are merged whatever their categories: the merged
// mask carries the one it overlaps most, and is not joined over function
// words to a neighbour of the other.
func TestMasksOverlapTwoCategories(t *testing.T) {
	text := "Il jeûne pendant le ramadan malgré son diabète et son asthme."
	at := func(s, typ string) indecis.Span {
		i := strings.Index(text, s)
		return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ}
	}
	found := []indecis.Span{
		at("jeûne pendant le ramadan", "QUASI"),
		at("ramadan malgré son diabète", "QUASI"),
		at("asthme", "QUASI"),
	}
	sensitive := []indecis.Span{at("jeûne pendant le ramadan", "BELIEF"), at("diabète", "HEALTH"), at("asthme", "HEALTH")}
	var got []string
	for _, m := range Masks(text, found, nil, sensitive) {
		got = append(got, text[m.Start:m.End]+" "+m.Sensitive)
	}
	want := []string{"jeûne pendant le ramadan malgré son diabète BELIEF", "asthme HEALTH"}
	if !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}

// Masks that touch, without a character in between, are joined as masks
// separated by function words: only within one category. They touch once
// refined: here, over the slash that ends the first one.
func TestMasksTouching(t *testing.T) {
	text := "Elle consulte pour son diabète/prière du vendredi."
	at := func(s, typ string) indecis.Span {
		i := strings.Index(text, s)
		return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ}
	}
	diabetes, prayer := at("diabète", "HEALTH"), at("prière du vendredi", "QUASI")
	diabetes.End++ // the slash
	found := []indecis.Span{diabetes, prayer}
	if diabetes.End != prayer.Start {
		t.Fatal("the masks do not touch")
	}
	for _, c := range []struct {
		sensitive []indecis.Span
		want      []string
	}{
		{[]indecis.Span{diabetes, at("prière du vendredi", "BELIEF")}, []string{"diabète/ HEALTH", "prière du vendredi BELIEF"}},
		{[]indecis.Span{diabetes, at("prière du vendredi", "HEALTH")}, []string{"diabète/prière du vendredi HEALTH"}},
	} {
		var got []string
		for _, m := range Masks(text, found, nil, c.sensitive) {
			got = append(got, text[m.Start:m.End]+" "+m.Sensitive)
		}
		if !slices.Equal(got, c.want) {
			t.Errorf("got %q, want %q", got, c.want)
		}
	}
}

// Between two sensitive passages that overlap a mask as much, the more
// confident wins, then the first in the text, whatever the order of the
// model's list.
func TestCategoryTie(t *testing.T) {
	text := "suivi pour un asthme au temple"
	at := func(s, typ string, conf float64) indecis.Span {
		i := strings.Index(text, s)
		return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ, Confidence: conf}
	}
	whole := Span{0, len(text)}
	for _, c := range []struct {
		sensitive []indecis.Span
		want      string
	}{
		{[]indecis.Span{at("asthme", "HEALTH", 0.6), at("temple", "BELIEF", 0.9)}, "BELIEF"},
		{[]indecis.Span{at("temple", "BELIEF", 0.9), at("asthme", "HEALTH", 0.6)}, "BELIEF"},
		{[]indecis.Span{at("temple", "BELIEF", 0.7), at("asthme", "HEALTH", 0.7)}, "HEALTH"},
		{[]indecis.Span{at("asthme", "HEALTH", 0.7), at("temple", "BELIEF", 0.7)}, "HEALTH"},
	} {
		if got := category(text, whole, c.sensitive); got != c.want {
			t.Errorf("%v: got %q, want %q", c.sensitive, got, c.want)
		}
	}
}

// OnFound gives a refined mask its own category, and an entity a filter
// adds the category of the sensitive passage it overlaps.
func TestFoundCategories(t *testing.T) {
	text := "Elle a une lombalgie chronique, voir le 06 12 34 56 78."
	at := func(s, typ string) indecis.Span {
		i := strings.Index(text, s)
		return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ, Confidence: 1}
	}
	sensitive := []indecis.Span{at("lombalgie chronique", "HEALTH")}
	phone := at("06 12 34 56 78", "CODE")
	r := &Recognizer{known: map[string][]decided{
		text: {{masks: append([]indecis.Span{at("une lombalgie chronique", "QUASI")}, sensitive...), sensitive: sensitive}},
	}}
	r.filters = []ner.EntityFilter{func(_ string, ents []ner.Entity) []ner.Entity {
		return append(ents, ner.Entity{Text: phone.Text, Type: "PHONE", Start: phone.Start, End: phone.End})
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
	want := []string{"une lombalgie chronique HEALTH", "06 12 34 56 78 "}
	if !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
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
