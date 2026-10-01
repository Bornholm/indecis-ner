package recognizer

import (
	"os"
	"strings"
	"testing"

	"github.com/bornholm/go-anon/pkg/anonymizer"
	"github.com/bornholm/go-anon/pkg/ner"
	"github.com/bornholm/indecis"
)

// model loads the model of INDECIS_NER_MODEL (default runs/a8m-full).
func model(t *testing.T) *indecis.Model {
	dir := os.Getenv("INDECIS_NER_MODEL")
	if dir == "" {
		dir = "../runs/a8m-full"
	}
	if _, err := os.Stat(dir + "/indecis.json"); err != nil {
		t.Skipf("no model in %s: set INDECIS_NER_MODEL", dir)
	}
	m, err := indecis.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// go-anon's anonymizer runs on the indecis recognizer, strict
// verification and identifiers by regular expression included.
func TestAnonymizeWithIndecis(t *testing.T) {
	r := New(model(t), "entities", ner.RegexEntityFilter(ner.BuiltinRegexPatterns))
	a := anonymizer.New(r, anonymizer.Config{Strategy: anonymizer.TagReplace, ConsistentMap: true})
	text := "Jean Dupont habite à Lyon depuis 2019. Écrivez-lui à jean.dupont@example.org ; Jean Dupont répondra."
	res, err := a.Anonymize(text, anonymizer.WithStrictVerification())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%s", res.Text)
	for _, leak := range []string{"Jean Dupont", "Lyon", "jean.dupont@example.org"} {
		if strings.Contains(res.Text, leak) {
			t.Errorf("%q left in %q", leak, res.Text)
		}
	}
	back, err := a.Deanonymize(res.Text, res.Mapping)
	if err != nil || back != text {
		t.Fatalf("deanonymized: %q, %v", back, err)
	}
}
