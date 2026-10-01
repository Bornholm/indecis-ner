package synth

import (
	"embed"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/bornholm/go-anon/pkg/synth/gazetteer"
	gtemplate "github.com/bornholm/go-anon/pkg/synth/template"
	"github.com/bornholm/go-anon/pkg/synth/value"
	"github.com/bornholm/indecis/dataset"
)

//go:embed lists/*.txt
var listFS embed.FS

// generators lists the placeholder generators. The first ones come from
// go-anon's value package: identities, organisations, addresses and
// identifiers with valid keys.
var generators = map[string]bool{
	"PER": true, "ORG": true, "ADDR": true, "email": true, "phone": true, "iban": true,
	"siret": true, "siren": true, "ref": true, "decoy": true,
	"first": true, "pick": true, "int": true, "nir": true, "matricule": true, "date": true, "money": true, "age": true,
}

// Span is a labeled passage of a generated document, in bytes.
type Span struct {
	Start, End           int
	Type, Mask, Sensitive string
}

// Doc is a generated document.
type Doc struct {
	Template, Type string
	Text           string
	Spans          []Span
}

// Example converts the document into an indecis example with the three
// questions, its template as family.
func (d Doc) Example() dataset.Example {
	ents, mask, sens := []any{}, []any{}, []any{}
	for _, s := range d.Spans {
		ents = append(ents, map[string]any{"start": s.Start, "end": s.End, "type": s.Type})
		if s.Mask != "" {
			mask = append(mask, map[string]any{"start": s.Start, "end": s.End, "type": s.Mask})
		}
		if s.Sensitive != "" {
			sens = append(sens, map[string]any{"start": s.Start, "end": s.End, "type": s.Sensitive})
		}
	}
	return dataset.Example{Text: d.Text, Labels: map[string]any{"entities": ents, "mask": mask, "sensitive": sens}, Family: d.Template}
}

// Lists holds the value lists of lists/*.txt, by name.
type Lists map[string][]string

// LoadLists reads the embedded lists: one value per line, # comments. A
// value "une |lombalgie chronique" writes its article outside the span.
func LoadLists() (Lists, error) {
	entries, err := listFS.ReadDir("lists")
	if err != nil {
		return nil, err
	}
	out := Lists{}
	for _, e := range entries {
		b, err := listFS.ReadFile("lists/" + e.Name())
		if err != nil {
			return nil, err
		}
		var vals []string
		for _, line := range strings.Split(string(b), "\n") {
			line = strings.TrimSpace(line)
			if line != "" && !strings.HasPrefix(line, "#") {
				vals = append(vals, line)
			}
		}
		out[strings.TrimSuffix(e.Name(), ".txt")] = vals
	}
	return out, nil
}

// Generate writes n documents, templates drawn by weight.
func Generate(templates []*Template, bundle *gazetteer.Bundle, lists Lists, n int, seed int64) ([]Doc, error) {
	rng := rand.New(rand.NewSource(seed))
	var total float64
	for _, t := range templates {
		total += t.Weight
	}
	docs := make([]Doc, 0, n)
	for i := 0; i < n; i++ {
		x := rng.Float64() * total
		t := templates[len(templates)-1]
		for _, c := range templates {
			if x < c.Weight {
				t = c
				break
			}
			x -= c.Weight
		}
		d, err := Render(t, bundle, lists, rng.Int63())
		if err != nil {
			return nil, fmt.Errorf("%s: %w", t.Name, err)
		}
		docs = append(docs, d)
	}
	return docs, nil
}

type renderer struct {
	t         *Template
	rng       *rand.Rand
	vg        *value.Generator
	lists     Lists
	slots     map[string]string
	dates     map[string]time.Time
	amounts   map[string]float64
	optionals map[string]bool
	base      time.Time
	b         strings.Builder
	spans     []Span
}

// Render writes one document.
func Render(t *Template, bundle *gazetteer.Bundle, lists Lists, seed int64) (Doc, error) {
	rng := rand.New(rand.NewSource(seed))
	r := &renderer{
		t: t, rng: rng, vg: value.New(rng, bundle), lists: lists,
		slots: map[string]string{}, dates: map[string]time.Time{}, amounts: map[string]float64{}, optionals: map[string]bool{},
		base: time.Date(2022+rng.Intn(4), time.Month(1+rng.Intn(12)), 1+rng.Intn(28), 0, 0, 0, 0, time.UTC),
	}
	if err := r.nodes(t.Body, 0); err != nil {
		return Doc{}, err
	}
	return Doc{Template: t.Name, Type: t.Type, Text: r.b.String(), Spans: r.spans}, nil
}

func (r *renderer) nodes(nodes []node, depth int) error {
	if depth > 8 {
		return fmt.Errorf("blocks nested too deep")
	}
	for _, n := range nodes {
		switch x := n.(type) {
		case text:
			r.b.WriteString(x.s)
		case pad:
			r.b.WriteString(strings.Repeat(" ", x.min+r.rng.Intn(x.max-x.min+1)))
		case optional:
			keep, seen := r.optionals[x.name]
			if !seen {
				keep = r.rng.Float64() < 0.6
				r.optionals[x.name] = keep
			}
			if keep {
				if err := r.nodes(x.body, depth+1); err != nil {
					return err
				}
			}
		case lines:
			for range x.min + r.rng.Intn(x.max-x.min+1) {
				if err := r.nodes(r.t.Blocks[x.block], depth+1); err != nil {
					return err
				}
			}
		case one:
			if err := r.nodes(r.t.Blocks[x.blocks[r.rng.Intn(len(x.blocks))]], depth+1); err != nil {
				return err
			}
		case placeholder:
			v, err := r.value(x)
			if err != nil {
				return err
			}
			v += x.args["suffix"] // "11" + " semaines": the unit inside the span
			// A list value "une|lombalgie" keeps its article out of the span.
			if before, after, ok := strings.Cut(v, "|"); ok && x.gen == "pick" {
				r.b.WriteString(before)
				v = after
			}
			start := r.b.Len()
			r.b.WriteString(v)
			if x.annotated {
				s, e := trimSpan(r.b.String(), start, r.b.Len())
				if s < e {
					r.spans = append(r.spans, Span{s, e, x.typ, x.mask, x.sensitive})
				}
			}
		}
	}
	return nil
}

func trimSpan(text string, s, e int) (int, int) {
	for s < e && unicode.IsSpace(rune(text[s])) {
		s++
	}
	for e > s && unicode.IsSpace(rune(text[e-1])) {
		e--
	}
	return s, e
}

func (r *renderer) memo(key, slot string, mk func() (string, error)) (string, error) {
	if slot == "" {
		return mk()
	}
	if v, ok := r.slots[key+":"+slot]; ok {
		return v, nil
	}
	v, err := mk()
	if err == nil {
		r.slots[key+":"+slot] = v
	}
	return v, err
}

func (r *renderer) intArg(p placeholder, name string, def int) int {
	if v, ok := p.args[name]; ok {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func (r *renderer) value(p placeholder) (string, error) {
	switch p.gen {
	case "first":
		// The first name of a PER slot, as an email is signed.
		segs := r.vg.Render(gtemplate.Placeholder{Kind: "PER", Slot: p.slot, Args: map[string]string{"form": "prenom_nom"}})
		if len(segs) == 0 || segs[0].Text == "" {
			return "", fmt.Errorf("first:%s: no person", p.slot)
		}
		return strings.Fields(segs[0].Text)[0], nil
	case "pick":
		list := r.lists[p.args["list"]]
		if len(list) == 0 {
			return "", fmt.Errorf("unknown or empty list %q", p.args["list"])
		}
		return r.memo("pick:"+p.args["list"], p.slot, func() (string, error) { return list[r.rng.Intn(len(list))], nil })
	case "int":
		lo, hi := r.intArg(p, "min", 0), r.intArg(p, "max", 10)
		return r.memo("int", p.slot, func() (string, error) { return strconv.Itoa(lo + r.rng.Intn(hi-lo+1)), nil })
	case "nir":
		return r.memo("nir", p.slot, func() (string, error) { return r.nir(p), nil })
	case "matricule":
		return r.memo("matricule", p.slot, func() (string, error) {
			digits := r.intArg(p, "digits", 6)
			var b strings.Builder
			b.WriteString(p.args["prefix"])
			for range digits {
				b.WriteByte(byte('0' + r.rng.Intn(10)))
			}
			return b.String(), nil
		})
	case "date":
		return formatDate(r.date(p.slot, p.args), p.args["format"]), nil
	case "age":
		birth := r.date(p.args["of"], map[string]string{"rel": "birth"})
		years := r.base.Year() - birth.Year()
		if r.base.YearDay() < birth.YearDay() {
			years--
		}
		return fmt.Sprintf("%d ans", years), nil
	case "money":
		return formatMoney(r.money(p), p.args["format"]), nil
	}
	// go-anon's generators.
	gp := gtemplate.Placeholder{Kind: p.gen, Slot: p.slot, Args: p.args}
	var b strings.Builder
	for _, seg := range r.vg.Render(gp) {
		b.WriteString(seg.Text)
	}
	if b.Len() == 0 {
		return "", fmt.Errorf("%s:%s renders nothing (arguments %v)", p.gen, p.slot, p.args)
	}
	return b.String(), nil
}

// nir draws a French social security number with its valid key, written
// with spaces or not.
func (r *renderer) nir(p placeholder) string {
	sex := 1 + r.rng.Intn(2)
	birth := r.date(p.args["birth"], map[string]string{"rel": "birth"})
	dept := 1 + r.rng.Intn(95)
	commune := 1 + r.rng.Intn(990)
	order := 1 + r.rng.Intn(999)
	body := fmt.Sprintf("%d%02d%02d%02d%03d%03d", sex, birth.Year()%100, int(birth.Month()), dept, commune, order)
	n, _ := strconv.ParseInt(body, 10, 64)
	key := 97 - n%97
	if p.args["format"] == "compact" {
		return fmt.Sprintf("%s%02d", body, key)
	}
	return fmt.Sprintf("%s %s %s %s %s %s %02d", body[:1], body[1:3], body[3:5], body[5:7], body[7:10], body[10:13], key)
}

// date draws or recalls the date of a slot. rel places it relative to the
// document's date: birth (an adult of working age), hire, past, recent,
// future, issue, month_start, month_end; after=slot adds days=N-M to
// another date.
func (r *renderer) date(slot string, args map[string]string) time.Time {
	if d, ok := r.dates[slot]; ok && slot != "" {
		return d
	}
	days := func(lo, hi int) int { return lo + r.rng.Intn(hi-lo+1) }
	b := r.base
	var d time.Time
	if after, ok := args["after"]; ok {
		lo, hi, err := parseRange(args["days"])
		if err != nil {
			lo, hi = 1, 30
		}
		d = r.date(after, map[string]string{"rel": "past"}).AddDate(0, 0, days(lo, hi))
	} else {
		switch args["rel"] {
		case "birth":
			d = b.AddDate(-days(19, 64), 0, -days(0, 364))
		case "hire":
			d = b.AddDate(-days(0, 25), 0, -days(0, 364))
			if birth, ok := r.dates["naissance"]; ok && d.Before(birth.AddDate(18, 0, 0)) {
				d = birth.AddDate(18+days(0, 5), 0, days(0, 300))
			}
		case "past":
			d = b.AddDate(0, 0, -days(30, 700))
		case "recent":
			d = b.AddDate(0, 0, -days(0, 30))
		case "future":
			d = b.AddDate(0, 0, days(1, 60))
		case "month_start":
			d = time.Date(b.Year(), b.Month(), 1, 0, 0, 0, 0, time.UTC)
		case "month_end":
			d = time.Date(b.Year(), b.Month()+1, 0, 0, 0, 0, 0, time.UTC)
		default: // issue
			d = b
		}
	}
	if slot != "" {
		r.dates[slot] = d
	}
	return d
}

var monthsFR = []string{"janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"}

func formatDate(d time.Time, format string) string {
	switch format {
	case "long":
		day := strconv.Itoa(d.Day())
		if d.Day() == 1 {
			day = "1er"
		}
		return fmt.Sprintf("%s %s %d", day, monthsFR[d.Month()-1], d.Year())
	case "month_year":
		return fmt.Sprintf("%s %d", monthsFR[d.Month()-1], d.Year())
	case "dash":
		return d.Format("02-01-2006")
	case "iso":
		return d.Format("2006-01-02")
	case "short":
		return d.Format("02/01/06")
	default:
		return d.Format("02/01/2006")
	}
}

// money draws an amount between min and max, or derives it from another
// slot: of=slot|ratio=0.78.
func (r *renderer) money(p placeholder) float64 {
	if v, ok := r.amounts[p.slot]; ok && p.slot != "" {
		return v
	}
	var v float64
	if of, ok := p.args["of"]; ok {
		ratio, _ := strconv.ParseFloat(p.args["ratio"], 64)
		v = r.amounts[of] * ratio
	} else {
		lo, hi := float64(r.intArg(p, "min", 0)), float64(r.intArg(p, "max", 100))
		v = lo + r.rng.Float64()*(hi-lo)
	}
	v = math.Round(v*100) / 100
	if p.slot != "" {
		r.amounts[p.slot] = v
	}
	return v
}

// formatMoney writes an amount the French way, "1 750,00", with " €" for
// format=eur.
func formatMoney(v float64, format string) string {
	s := strconv.FormatFloat(v, 'f', 2, 64)
	intPart, dec, _ := strings.Cut(s, ".")
	var b strings.Builder
	for i, c := range intPart {
		if i > 0 && (len(intPart)-i)%3 == 0 {
			b.WriteByte(' ')
		}
		b.WriteRune(c)
	}
	out := b.String() + "," + dec
	if format == "eur" {
		out += " €"
	}
	return out
}
