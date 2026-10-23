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
// and the punctuation of sentences at the edges are left out.
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
		{at("diabète, "), "diabète"},
		{at(", avant"), "avant"},
		{at("abète, av"), "diabète, avant"},
		// A cut inside a word that holds a two-byte rune, "ièr", and a cut
		// inside that rune, "è" being bytes 3 and 4 of "prière".
		{Span{at("prière").Start + 2, at("prière").End - 1}, "prière"},
		{Span{at("prière").Start + 4, at("prière").Start + 4 + 1}, "prière"},
		{Span{at("prière").Start, at("prière").Start + 4}, "prière"},
		{at("son diabète, avant"), "son diabète, avant"},
		// A bracket without its partner is left out, not one with it.
		{at("diabète, avant la prière du vendredi."), "diabète, avant la prière du vendredi"},
	} {
		got := Refine(text, []Span{c.in}, nil)
		if len(got) != 1 || text[got[0].Start:got[0].End] != c.want {
			t.Errorf("%q: got %v, want %q", text[c.in.Start:c.in.End], got, c.want)
		}
	}
	// An empty mask, even inside a word or a rune, cuts none and is
	// dropped.
	for _, i := range []int{at("prière").Start + 2, at("prière").Start + 4} {
		if got := Refine(text, []Span{{i, i}}, nil); len(got) != 0 {
			t.Errorf("empty mask at %d: got %v, want none", i, got)
		}
	}
	// A sign between two letters or digits at the edge cuts a word or a
	// number; a sign before a space does not.
	text = "Des rendez-vous à l'hôpital, né le 12/03/1980 (1,75 m)."
	for _, c := range []struct {
		in   Span
		want string
	}{
		{at("Des rendez-"), "Des rendez-vous"},
		{at("-vous à"), "rendez-vous à"},
		{at("'hôpital"), "l'hôpital"},
		{at("/03/1980"), "12/03/1980"},
		{at("12/03/"), "12/03/1980"},
		// Past the edge, the mask grows over the signs between letters or
		// digits too.
		{at("/1980"), "12/03/1980"},
		{at("12/03"), "12/03/1980"},
		{at("vous"), "rendez-vous"},
		{at("l'"), "l'hôpital"},
		{at(",75 m"), "1,75 m"},
		{at("hôpital, "), "l'hôpital"},
		{at("1,75 m)"), "1,75 m"},
		{at("(1,75 m)"), "(1,75 m)"},
	} {
		got := Refine(text, []Span{c.in}, nil)
		if len(got) != 1 || text[got[0].Start:got[0].End] != c.want {
			t.Errorf("%q: got %v, want %q", text[c.in.Start:c.in.End], got, c.want)
		}
	}
	// A bracket at an edge is paired by depth: the partner of another
	// pair in the mask does not keep it.
	text = "Voir (annexe 2) (pièce jointe) et « dossier (suite) »."
	for _, c := range []struct {
		in   Span
		want string
	}{
		{at("(annexe 2) ("), "(annexe 2)"},
		{at(") (pièce jointe)"), "(pièce jointe)"},
		{at("(annexe 2) (pièce jointe)"), "(annexe 2) (pièce jointe)"},
		{at("2) (pièce jointe)"), "2) (pièce jointe)"},
		{at("« dossier (suite)"), "dossier (suite)"},
		{at("« dossier (suite) »"), "« dossier (suite) »"},
		{at("(suite) »"), "(suite)"},
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

// A passage the model gives whole over two identifiers, or two names,
// is cut into one mask each: go-anon gives each its own placeholder. The
// full stop after a number is left out.
func TestMasksSplit(t *testing.T) {
	text := "Coordonnées : 06 42 18 77 05, nadia.ferrand@laposte.net, 8 allée des Peupliers. NIR : 2 83 11 71 076 112 47. " +
		"Présents : Mme Carmen Roussel et M. Paul Martin, chef de l'équipe de nuit. Dr Weiss, 42 ans."
	at := func(s, typ string) indecis.Span {
		i := strings.Index(text, s)
		if i < 0 {
			t.Fatalf("%q not in text", s)
		}
		return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ}
	}
	found := []indecis.Span{
		at("06 42 18 77 05, nadia.ferrand@laposte.net", "DIRECT"),
		at("8 allée des Peupliers", "DIRECT"),
		at("2 83 11 71 076 112 47.", "DIRECT"),
		at("Mme Carmen Roussel et M. Paul Martin, chef de l'équipe de nuit", "DIRECT"),
		at("Dr Weiss, 42 ans", "DIRECT"),
	}
	typed := []indecis.Span{
		at("06 42 18 77 05", "CODE"), at("nadia.ferrand@laposte.net", "CODE"), at("8 allée des Peupliers", "LOC"),
		at("2 83 11 71 076 112 47", "CODE"),
		at("Carmen Roussel", "PERSON"), at("Paul Martin", "PERSON"), at("chef de l'équipe de nuit", "DEM"),
		at("Weiss", "PERSON"), at("42 ans", "DEM"),
	}
	var got []string
	for _, m := range Masks(text, found, typed, nil) {
		got = append(got, text[m.Start:m.End]+" "+string(m.Type))
	}
	want := []string{
		"06 42 18 77 05 CODE", "nadia.ferrand@laposte.net CODE", "8 allée des Peupliers LOC",
		"2 83 11 71 076 112 47 CODE",
		// A civility stays with its name, "et" is left readable, and what
		// follows the last name is a mask of its own. A mask with one name
		// only is not cut.
		"Mme Carmen Roussel PERSON", "M. Paul Martin PERSON", "chef de l'équipe de nuit DEM",
		"Dr Weiss, 42 ans DEM",
	}
	if !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}

// The shapes the cut depends on: the passages of the entities question
// do not always bound a name as the text does.
func TestMasksSplitShapes(t *testing.T) {
	type typedSpan struct{ text, typ string }
	cases := []struct {
		name, text, found string
		typed             []typedSpan
		want              []string
	}{
		{"one passage over identifiers and an address",
			"Suivi : 06 42 18 77 05, nadia.ferrand@laposte.net, 8 allée des Peupliers.",
			"06 42 18 77 05, nadia.ferrand@laposte.net, 8 allée des Peupliers.",
			[]typedSpan{{"06 42 18 77 05", "CODE"}, {"nadia.ferrand@laposte.net", "CODE"}, {"8 allée des Peupliers", "LOC"}},
			[]string{"06 42 18 77 05 CODE", "nadia.ferrand@laposte.net CODE", "8 allée des Peupliers LOC"}},
		// The pieces do not grow over the sign back into one mask.
		{"two names separated by a sign",
			"Mme Carmen Roussel/M. Paul Martin sont venus.", "Mme Carmen Roussel/M. Paul Martin",
			[]typedSpan{{"Carmen Roussel", "PERSON"}, {"Paul Martin", "PERSON"}},
			[]string{"Mme Carmen Roussel PERSON", "M. Paul Martin PERSON"}},
		{"one name in two passages that touch",
			"Mme Carmen Roussel vient.", "Mme Carmen Roussel",
			[]typedSpan{{"Carmen", "PERSON"}, {" Roussel", "PERSON"}},
			[]string{"Mme Carmen Roussel PERSON"}},
		{"one name in two passages, a space between them",
			"Mme Carmen Roussel vient.", "Mme Carmen Roussel",
			[]typedSpan{{"Carmen", "PERSON"}, {"Roussel", "PERSON"}},
			[]string{"Mme Carmen Roussel PERSON"}},
		{"one identifier in two passages, a sign between them",
			"Écrire à nadia.ferrand@laposte.net demain.", "nadia.ferrand@laposte.net",
			[]typedSpan{{"nadia", "CODE"}, {"ferrand@laposte.net", "CODE"}},
			[]string{"nadia.ferrand@laposte.net CODE"}},
		{"a passage cut inside a word",
			"Mme Carmen Roussel et M. Paul Martin.", "Mme Carmen Roussel et M. Paul Martin",
			[]typedSpan{{"Carmen Rouss", "PERSON"}, {"aul Martin", "PERSON"}},
			[]string{"Mme Carmen Roussel PERSON", "M. Paul Martin PERSON"}},
		// A name the mask only clips is not a name of its own.
		{"a name the mask clips",
			"Vu par Carmen Roussel avec Paul Martin.", "Carmen Roussel avec P",
			[]typedSpan{{"Carmen Roussel", "PERSON"}, {"Paul Martin", "PERSON"}},
			[]string{"Carmen Roussel avec Paul PERSON"}},
		// Only the civility goes with the next name: an apposition is a
		// mask of its own.
		{"an apposition between two names",
			"Mme Roussel, présidente, et M. Paul Martin.", "Mme Roussel, présidente, et M. Paul Martin",
			[]typedSpan{{"Roussel", "PERSON"}, {"Paul Martin", "PERSON"}},
			[]string{"Mme Roussel PERSON", "présidente MISC", "M. Paul Martin PERSON"}},
		{"a sentence between two names",
			"Mme Roux. Elle est suivie au CMP, M. Paul Martin.", "Mme Roux. Elle est suivie au CMP, M. Paul Martin",
			[]typedSpan{{"Roux", "PERSON"}, {"Paul Martin", "PERSON"}},
			[]string{"Mme Roux PERSON", "est suivie au CMP MISC", "M. Paul Martin PERSON"}},
		// A civility is a title: the word before a full stop is not one.
		{"a full stop before a civility",
			"Mme Roux. Elle est suivie au CMP. M. Paul Martin.", "Mme Roux. Elle est suivie au CMP. M. Paul Martin",
			[]typedSpan{{"Roux", "PERSON"}, {"Paul Martin", "PERSON"}},
			[]string{"Mme Roux PERSON", "est suivie au CMP MISC", "M. Paul Martin PERSON"}},
		{"an apposition right before a civility",
			"Mme Roussel, présidente M. Paul Martin.", "Mme Roussel, présidente M. Paul Martin",
			[]typedSpan{{"Roussel", "PERSON"}, {"Paul Martin", "PERSON"}},
			[]string{"Mme Roussel PERSON", "présidente MISC", "M. Paul Martin PERSON"}},
		{"two titles",
			"Mme Roussel et Mme Dr Weiss.", "Mme Roussel et Mme Dr Weiss",
			[]typedSpan{{"Roussel", "PERSON"}, {"Weiss", "PERSON"}},
			[]string{"Mme Roussel PERSON", "Mme Dr Weiss PERSON"}},
		{"two names separated by a comma without a space",
			"Contacts : Roussel,Martin.", "Roussel,Martin",
			[]typedSpan{{"Roussel", "PERSON"}, {"Martin", "PERSON"}},
			[]string{"Roussel PERSON", "Martin PERSON"}},
		{"two numbers separated by a slash",
			"Tél. 06 42 18 77 05/06 12 34 56 78.", "06 42 18 77 05/06 12 34 56 78",
			[]typedSpan{{"06 42 18 77 05", "CODE"}, {"06 12 34 56 78", "CODE"}},
			[]string{"06 42 18 77 05 CODE", "06 12 34 56 78 CODE"}},
		{"two names on two lines",
			"Mme Roussel\nM. Paul Martin", "Mme Roussel\nM. Paul Martin",
			[]typedSpan{{"Roussel", "PERSON"}, {"Paul Martin", "PERSON"}},
			[]string{"Mme Roussel PERSON", "M. Paul Martin PERSON"}},
		{"a whole title before a full stop",
			"Mme Roux a vu le docteur. M. Paul Martin.", "Mme Roux a vu le docteur. M. Paul Martin",
			[]typedSpan{{"Roux", "PERSON"}, {"Paul Martin", "PERSON"}},
			[]string{"Mme Roux PERSON", "vu le docteur MISC", "M. Paul Martin PERSON"}},
		{"function words after the last name",
			"Mme Roussel et M. Martin et le service de nuit.", "Mme Roussel et M. Martin et le service de nuit",
			[]typedSpan{{"Roussel", "PERSON"}, {"Martin", "PERSON"}},
			[]string{"Mme Roussel PERSON", "M. Martin PERSON", "service de nuit MISC"}},
		// A one-letter title is written with a capital: "m." ends a
		// measure.
		{"a unit before a full stop",
			"Mme Roux mesure 1,75 m. Paul Martin.", "Mme Roux mesure 1,75 m. Paul Martin",
			[]typedSpan{{"Roux", "PERSON"}, {"Paul Martin", "PERSON"}},
			[]string{"Mme Roux PERSON", "mesure 1,75 m MISC", "Paul Martin PERSON"}},
		{"a name and a number separated by a space",
			"Joindre M. Paul Martin 06 42 18 77 05.", "M. Paul Martin 06 42 18 77 05",
			[]typedSpan{{"Paul Martin", "PERSON"}, {"06 42 18 77 05", "CODE"}},
			[]string{"M. Paul Martin PERSON", "06 42 18 77 05 CODE"}},
		// A name keeps a last word that is a function word, and is kept
		// when it holds nothing else.
		{"a name made of a function word",
			"Carmen Roussel et Le sont venus.", "Carmen Roussel et Le",
			[]typedSpan{{"Carmen Roussel", "PERSON"}, {"Le", "PERSON"}},
			[]string{"Carmen Roussel PERSON", "Le PERSON"}},
		{"a name ending on a function word",
			"Jean Le et Paul Des sont venus.", "Jean Le et Paul Des",
			[]typedSpan{{"Jean Le", "PERSON"}, {"Paul Des", "PERSON"}},
			[]string{"Jean Le PERSON", "Paul Des PERSON"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			at := func(s, typ string) indecis.Span {
				i := strings.Index(c.text, s)
				if i < 0 {
					t.Fatalf("%q not in text", s)
				}
				return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ}
			}
			found := []indecis.Span{at(c.found, "DIRECT")}
			var typed []indecis.Span
			for _, ts := range c.typed {
				typed = append(typed, at(ts.text, ts.typ))
			}
			var got []string
			for _, m := range Masks(c.text, found, typed, nil) {
				got = append(got, c.text[m.Start:m.End]+" "+string(m.Type))
			}
			if !slices.Equal(got, c.want) {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}

// Each piece of a cut mask takes the category of the sensitive passage
// it overlaps, none if it overlaps none.
func TestMasksSplitSensitive(t *testing.T) {
	text := "Mme Roussel, diabétique, et M. Paul Martin sont venus."
	at := func(s, typ string) indecis.Span {
		i := strings.Index(text, s)
		return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ}
	}
	found := []indecis.Span{at("Mme Roussel, diabétique, et M. Paul Martin", "DIRECT")}
	typed := []indecis.Span{at("Roussel", "PERSON"), at("Paul Martin", "PERSON")}
	sensitive := []indecis.Span{at("diabétique", "HEALTH")}
	var got []string
	for _, m := range Masks(text, found, typed, sensitive) {
		got = append(got, text[m.Start:m.End]+" "+m.Sensitive)
	}
	want := []string{"Mme Roussel ", "diabétique HEALTH", "M. Paul Martin "}
	if !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}

// Masks between brackets that touch keep their brackets, each pair
// being whole: they still touch, and are joined within one category
// only, as other masks that touch.
func TestMasksTouchingBrackets(t *testing.T) {
	text := "Elle consulte (son diabète)(prière du vendredi)."
	at := func(s, typ string) indecis.Span {
		i := strings.Index(text, s)
		return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ}
	}
	found := []indecis.Span{at("(son diabète)", "QUASI"), at("(prière du vendredi)", "QUASI")}
	for _, c := range []struct {
		prayer string
		want   []string
	}{
		{"BELIEF", []string{"(son diabète) HEALTH", "(prière du vendredi) BELIEF"}},
		{"HEALTH", []string{"(son diabète)(prière du vendredi) HEALTH"}},
	} {
		sensitive := []indecis.Span{at("diabète", "HEALTH"), at("prière du vendredi", c.prayer)}
		var got []string
		for _, m := range Masks(text, found, nil, sensitive) {
			got = append(got, text[m.Start:m.End]+" "+m.Sensitive)
		}
		if !slices.Equal(got, c.want) {
			t.Errorf("got %q, want %q", got, c.want)
		}
	}
}

// Across a comma, masks of different types are kept apart: an
// organisation and its address, a post and its service. Masks of one
// type, or of no type, are joined, and so are masks of different types
// across function words alone.
func TestMasksListedTypes(t *testing.T) {
	text := "Établissement : Clinique du Val de Saône, 14 rue des Tanneurs, 71100 Chalon-sur-Saône. " +
		"Poste : infirmière de nuit, service de pneumologie. Il est magasinier sur le site de Stains, dans l'entrepôt, en équipe."
	at := func(s, typ string) indecis.Span {
		i := strings.Index(text, s)
		if i < 0 {
			t.Fatalf("%q not in text", s)
		}
		return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ}
	}
	found := []indecis.Span{
		at("Clinique du Val de Saône", "QUASI"), at("14 rue des Tanneurs", "QUASI"), at("71100 Chalon-sur-Saône", "QUASI"),
		at("infirmière de nuit", "QUASI"), at("service de pneumologie", "QUASI"),
		at("magasinier", "QUASI"), at("site de Stains", "QUASI"), at("entrepôt", "QUASI"),
	}
	typed := []indecis.Span{
		at("Clinique du Val de Saône", "ORG"), at("14 rue des Tanneurs", "LOC"), at("71100 Chalon-sur-Saône", "LOC"),
		at("infirmière de nuit", "DEM"), at("service de pneumologie", "ORG"),
		at("magasinier", "DEM"), at("site de Stains", "LOC"),
	}
	var got []string
	for _, m := range Masks(text, found, typed, nil) {
		got = append(got, text[m.Start:m.End]+" "+string(m.Type))
	}
	want := []string{
		"Clinique du Val de Saône ORG", "14 rue des Tanneurs, 71100 Chalon-sur-Saône LOC",
		"infirmière de nuit DEM", "service de pneumologie ORG",
		"magasinier sur le site de Stains, dans l'entrepôt LOC",
	}
	if !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}

// The types compared across a comma are those of the items next to it.
func TestMasksListedShapes(t *testing.T) {
	type typedSpan struct{ text, typ string }
	cases := []struct {
		name, text string
		found      []string
		typed      []typedSpan
		sensitive  []typedSpan
		want       []string
	}{
		{"an item typed MISC joins",
			"Clinique du Val de Saône, entrepôt nord.", []string{"Clinique du Val de Saône", "entrepôt nord"},
			[]typedSpan{{"Clinique du Val de Saône", "ORG"}, {"entrepôt nord", "MISC"}},
			nil,
			[]string{"Clinique du Val de Saône, entrepôt nord ORG"}},
		{"a comma and function words",
			"Clinique du Val de Saône, dans le quartier Saint-Jean.", []string{"Clinique du Val de Saône", "quartier Saint-Jean"},
			[]typedSpan{{"Clinique du Val de Saône", "ORG"}, {"quartier Saint-Jean", "LOC"}},
			nil,
			[]string{"Clinique du Val de Saône ORG", "quartier Saint-Jean LOC"}},
		// The left mask is typed after its description, DEM; the item
		// before the comma is an organisation, as the one after it.
		{"a description before the comma",
			"Infirmière de nuit en réanimation à l'hôpital Nord, service de pneumologie.",
			[]string{"Infirmière de nuit en réanimation", "hôpital Nord", "service de pneumologie"},
			[]typedSpan{{"Infirmière de nuit en réanimation", "DEM"}, {"hôpital Nord", "ORG"}, {"service de pneumologie", "ORG"}},
			nil,
			[]string{"Infirmière de nuit en réanimation à l'hôpital Nord, service de pneumologie DEM"}},
		{"another type before the comma",
			"Infirmière de nuit en réanimation à l'hôpital Nord, 14 rue des Tanneurs.",
			[]string{"Infirmière de nuit en réanimation", "hôpital Nord", "14 rue des Tanneurs"},
			[]typedSpan{{"Infirmière de nuit en réanimation", "DEM"}, {"hôpital Nord", "ORG"}, {"14 rue des Tanneurs", "LOC"}},
			nil,
			[]string{"Infirmière de nuit en réanimation à l'hôpital Nord DEM", "14 rue des Tanneurs LOC"}},
		// Words of no typed passage at the comma: the item has no type,
		// and joins.
		{"an untyped item at the comma",
			"Infirmière de nuit à l'hôpital, service de pneumologie.",
			[]string{"Infirmière de nuit à l'hôpital", "service de pneumologie"},
			[]typedSpan{{"Infirmière", "DEM"}, {"service de pneumologie", "ORG"}},
			nil,
			[]string{"Infirmière de nuit à l'hôpital, service de pneumologie ORG"}},
		// A passage that runs past the comma belongs to the item before it.
		{"a typed passage past the comma",
			"Clinique du Val de Saône, Chalon-sur-Saône.", []string{"Clinique du Val de Saône", "Chalon-sur-Saône"},
			[]typedSpan{{"Clinique du Val de Saône, Ch", "ORG"}, {"Chalon-sur-Saône", "LOC"}},
			nil,
			[]string{"Clinique du Val de Saône ORG", "Chalon-sur-Saône LOC"}},
		// Passages that end together are ranked whatever their order.
		{"passages ending together",
			"Hôpital Nord, 14 rue des Tanneurs.", []string{"Hôpital Nord", "14 rue des Tanneurs"},
			[]typedSpan{{"Nord", "LOC"}, {"Hôpital Nord", "ORG"}, {"14 rue des Tanneurs", "LOC"}},
			nil,
			[]string{"Hôpital Nord ORG", "14 rue des Tanneurs LOC"}},
		{"passages ending together, the other order",
			"Hôpital Nord, 14 rue des Tanneurs.", []string{"Hôpital Nord", "14 rue des Tanneurs"},
			[]typedSpan{{"Hôpital Nord", "ORG"}, {"Nord", "LOC"}, {"14 rue des Tanneurs", "LOC"}},
			nil,
			[]string{"Hôpital Nord ORG", "14 rue des Tanneurs LOC"}},
		// Of one type, masks of different categories are still kept apart.
		{"one type, two categories",
			"Il jeûne pendant le ramadan, diabétique depuis l'enfance.",
			[]string{"jeûne pendant le ramadan", "diabétique depuis l'enfance"},
			[]typedSpan{{"jeûne pendant le ramadan", "DEM"}, {"diabétique depuis l'enfance", "DEM"}},
			[]typedSpan{{"jeûne pendant le ramadan", "BELIEF"}, {"diabétique", "HEALTH"}},
			[]string{"jeûne pendant le ramadan DEM", "diabétique depuis l'enfance DEM"}},
		{"one type, one category",
			"Il jeûne pendant le ramadan, prie le vendredi.",
			[]string{"jeûne pendant le ramadan", "prie le vendredi"},
			[]typedSpan{{"jeûne pendant le ramadan", "DEM"}, {"prie le vendredi", "DEM"}},
			[]typedSpan{{"jeûne pendant le ramadan", "BELIEF"}, {"prie le vendredi", "BELIEF"}},
			[]string{"jeûne pendant le ramadan, prie le vendredi DEM"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			at := func(s, typ string) indecis.Span {
				i := strings.Index(c.text, s)
				if i < 0 {
					t.Fatalf("%q not in text", s)
				}
				return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ}
			}
			var found, typed []indecis.Span
			for _, f := range c.found {
				found = append(found, at(f, "QUASI"))
			}
			for _, ts := range c.typed {
				typed = append(typed, at(ts.text, ts.typ))
			}
			var sensitive []indecis.Span
			for _, ts := range c.sensitive {
				sensitive = append(sensitive, at(ts.text, ts.typ))
			}
			var got []string
			for _, m := range Masks(c.text, found, typed, sensitive) {
				got = append(got, c.text[m.Start:m.End]+" "+string(m.Type))
			}
			if !slices.Equal(got, c.want) {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}

// A semicolon, a colon or an ellipsis is left out of a mask's edge, and
// keeps two masks apart; a comma is left out too, but joins them.
func TestRefinePunctuationJoin(t *testing.T) {
	text := "Il a du diabète; asthme aussi. Il a du diabète, asthme aussi. Il a du diabète… et asthme."
	semi := strings.Index(text, "diabète;")
	comma := strings.LastIndex(text, "diabète,")
	dots := strings.LastIndex(text, "diabète…")
	spans := []Span{
		{semi, semi + len("diabète;")}, {semi + len("diabète; "), semi + len("diabète; asthme")},
		{comma, comma + len("diabète,")}, {comma + len("diabète, "), comma + len("diabète, asthme")},
		{dots, dots + len("diabète…")}, {dots + len("diabète… et "), dots + len("diabète… et asthme")},
	}
	var got []string
	for _, s := range Refine(text, spans, nil) {
		got = append(got, text[s.Start:s.End])
	}
	want := []string{"diabète", "asthme", "diabète, asthme", "diabète", "asthme"}
	if !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}

// A mask grown over a sign between two words takes the word of another
// mask: the two overlap and are merged whatever their categories, the
// loss of #2 that remains at such a sign.
func TestMasksJoiningSign(t *testing.T) {
	text := "Il a des rendez-vous médicaux le vendredi."
	at := func(s, typ string) indecis.Span {
		i := strings.Index(text, s)
		return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ}
	}
	found := []indecis.Span{at("rendez-", "QUASI"), at("vous médicaux", "QUASI")}
	sensitive := []indecis.Span{at("rendez-", "BELIEF"), at("vous médicaux", "HEALTH")}
	var got []string
	for _, m := range Masks(text, found, nil, sensitive) {
		got = append(got, text[m.Start:m.End]+" "+m.Sensitive)
	}
	want := []string{"rendez-vous médicaux HEALTH"}
	if !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}

// Masks that touch, without a character in between, are joined as masks
// separated by function words: only within one category. They touch once
// refined: here, over a slash after a space, which joins no word.
func TestMasksTouching(t *testing.T) {
	text := "Elle consulte pour son diabète /prière du vendredi."
	at := func(s, typ string) indecis.Span {
		i := strings.Index(text, s)
		return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ}
	}
	diabetes, prayer := at("diabète /", "HEALTH"), at("prière du vendredi", "QUASI")
	found := []indecis.Span{diabetes, prayer}
	if diabetes.End != prayer.Start {
		t.Fatal("the masks do not touch")
	}
	for _, c := range []struct {
		sensitive []indecis.Span
		want      []string
	}{
		{[]indecis.Span{diabetes, at("prière du vendredi", "BELIEF")}, []string{"diabète / HEALTH", "prière du vendredi BELIEF"}},
		{[]indecis.Span{diabetes, at("prière du vendredi", "HEALTH")}, []string{"diabète /prière du vendredi HEALTH"}},
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

// A typed passage under MinConfidence does not cut a mask.
func TestMinConfidenceTyped(t *testing.T) {
	text := "Contact : Carmen Roussel, Paul Martin."
	at := func(s, typ string, conf float64) indecis.Span {
		i := strings.Index(text, s)
		return indecis.Span{Start: i, End: i + len(s), Text: s, Type: typ, Confidence: conf}
	}
	masks := []indecis.Span{at("Carmen Roussel, Paul Martin", "DIRECT", 0.9)}
	for _, c := range []struct {
		conf float64
		want []string
	}{
		{0.2, []string{"Carmen Roussel, Paul Martin"}},
		{0.8, []string{"Carmen Roussel", "Paul Martin"}},
	} {
		typed := []indecis.Span{at("Carmen Roussel", "PERSON", 0.9), at("Paul Martin", "PERSON", c.conf)}
		r := &Recognizer{MinConfidence: 0.5, known: map[string][]decided{
			text: {{masks: masks, typed: typed}},
		}}
		var got []string
		r.OnFound = func(_ string, found []Found) {
			for _, f := range found {
				got = append(got, f.Text)
			}
		}
		if _, err := r.Recognize(text); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, c.want) {
			t.Errorf("typed confidence %g: got %q, want %q", c.conf, got, c.want)
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
