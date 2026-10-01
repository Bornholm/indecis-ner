// Package synth writes synthetic workplace documents whose personal data
// are labeled by construction, following GUIDELINES.md: a type (entities),
// a masking decision (mask) and, for special categories, a sensitive label.
//
// A template is the extracted text of a document, with placeholders:
//
//	type: bulletin_paie
//	weight: 1
//	---
//	Salarié : {{PER:salarie|form=civ_prenom_nom @PERSON/DIRECT}}
//	Emploi : {{pick:poste|list=postes @DEM/QUASI}}
//	Motif : {{pick:motif|list=pathologies @DEM/QUASI/HEALTH}}
//	Date d'émission : {{date:emission|rel=issue|format=slash}}
//
// The argument suffix=… is appended to any value, inside its span. The
// annotation after @ is TYPE/MASK[/SENSITIVE], MASK being DIRECT, QUASI
// or - (typed, not masked). A placeholder without @ is plain text. The slot
// (after the colon) keeps a value across the document: two placeholders of
// the same generator and slot render the same person, the same date.
//
// Other constructs: {{pad:N-M}} (N to M spaces), [?name]…[/] (an optional
// section, kept or dropped as a whole, the same decision for every section
// of that name), blocks defined between "@block name" and "@end" lines,
// {{LINES:name:N-M}} (the block repeated N to M times) and {{ONE:a,b,c}}
// (one of the blocks, drawn at random).
package synth

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	gtemplate "github.com/bornholm/go-anon/pkg/synth/template"
)

// Template is a parsed document template.
type Template struct {
	Name   string
	Type   string
	Weight float64
	Body   []node
	Blocks map[string][]node
}

type node interface{}

type text struct{ s string }

// placeholder is a value to generate, with its labels.
type placeholder struct {
	gen, slot string
	args      map[string]string
	typ, mask string // mask: DIRECT, QUASI or "" (typed only)
	sensitive string
	annotated bool
}

type pad struct{ min, max int }

type lines struct {
	block    string
	min, max int
}

type one struct{ blocks []string }

type optional struct {
	name string
	body []node
}

// personForms are the forms of go-anon's PER generator, couple excepted:
// it writes two people as one string.
var personForms = map[string]bool{"civ_prenom_nom": true, "civ_nom_prenom": true, "prenom_nom": true, "prenom_nom_caps": true,
	"nom_prenom_caps": true, "nom_seul": true, "titre_nom": true, "titre_nom_caps": true, "nom_naissance": true, "nom_prenom_etiquette": true}

var (
	tokenRe    = regexp.MustCompile(`\{\{.*?\}\}|\[\?[a-z0-9_]+\]|\[/\]`)
	types      = map[string]bool{"PERSON": true, "CODE": true, "LOC": true, "ORG": true, "DEM": true, "DATETIME": true, "QUANTITY": true, "MISC": true}
	masks      = map[string]bool{"DIRECT": true, "QUASI": true, "-": true}
	sensitives = map[string]bool{"HEALTH": true, "UNION": true, "BELIEF": true, "POLITICS": true, "ORIGIN": true, "SEXUALITY": true, "OFFENCE": true}
)

// LoadTemplates reads the *.tmpl files of a directory.
func LoadTemplates(dir string) ([]*Template, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*.tmpl"))
	if err != nil {
		return nil, err
	}
	sort.Strings(paths)
	var out []*Template
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		t, err := Parse(strings.TrimSuffix(filepath.Base(p), ".tmpl"), string(b))
		if err != nil {
			return nil, fmt.Errorf("%s: %w", p, err)
		}
		out = append(out, t)
	}
	return out, nil
}

// Parse reads a template.
func Parse(name, src string) (*Template, error) {
	head, body, ok := strings.Cut(src, "\n---\n")
	if !ok {
		return nil, fmt.Errorf("no header (--- line)")
	}
	t := &Template{Name: name, Weight: 1, Blocks: map[string][]node{}}
	sc := bufio.NewScanner(strings.NewReader(head))
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		v = strings.TrimSpace(v)
		switch strings.TrimSpace(k) {
		case "type":
			t.Type = v
		case "weight":
			w, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return nil, fmt.Errorf("weight %q", v)
			}
			t.Weight = w
		}
	}
	// Blocks first: "@block name" ... "@end", on lines of their own.
	var main strings.Builder
	var block string
	var cur strings.Builder
	for _, line := range strings.SplitAfter(body, "\n") {
		trimmed := strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(trimmed, "@block "):
			block = strings.TrimSpace(strings.TrimPrefix(trimmed, "@block "))
			cur.Reset()
		case trimmed == "@end" && block != "":
			// The block's last line break belongs to the line that uses it.
			nodes, err := parseNodes(strings.TrimSuffix(cur.String(), "\n"))
			if err != nil {
				return nil, fmt.Errorf("block %s: %w", block, err)
			}
			t.Blocks[block] = nodes
			block = ""
		case block != "":
			cur.WriteString(line)
		default:
			main.WriteString(line)
		}
	}
	if block != "" {
		return nil, fmt.Errorf("block %s not closed", block)
	}
	nodes, err := parseNodes(main.String())
	if err != nil {
		return nil, err
	}
	t.Body = nodes
	return t, t.check(t.Body)
}

func (t *Template) check(nodes []node) error {
	for _, n := range nodes {
		switch x := n.(type) {
		case lines:
			if _, ok := t.Blocks[x.block]; !ok {
				return fmt.Errorf("unknown block %q", x.block)
			}
		case one:
			for _, b := range x.blocks {
				if _, ok := t.Blocks[b]; !ok {
					return fmt.Errorf("unknown block %q", b)
				}
			}
		case optional:
			if err := t.check(x.body); err != nil {
				return err
			}
		}
	}
	for name, b := range t.Blocks {
		for _, n := range b {
			if l, ok := n.(lines); ok && l.block == name {
				return fmt.Errorf("block %s repeats itself", name)
			}
		}
	}
	return nil
}

func parseNodes(src string) ([]node, error) {
	var stack [][]node
	var names []string
	var cur []node
	last := 0
	for _, loc := range tokenRe.FindAllStringIndex(src, -1) {
		if loc[0] > last {
			cur = append(cur, text{src[last:loc[0]]})
		}
		last = loc[1]
		tok := src[loc[0]:loc[1]]
		switch {
		case strings.HasPrefix(tok, "[?"):
			stack = append(stack, cur)
			names = append(names, tok[2:len(tok)-1])
			cur = nil
		case tok == "[/]":
			if len(stack) == 0 {
				return nil, fmt.Errorf("[/] without [?name]")
			}
			opt := optional{names[len(names)-1], cur}
			cur = append(stack[len(stack)-1], opt)
			stack, names = stack[:len(stack)-1], names[:len(names)-1]
		default:
			n, err := parsePlaceholder(tok[2 : len(tok)-2])
			if err != nil {
				return nil, fmt.Errorf("%s: %w", tok, err)
			}
			cur = append(cur, n)
		}
	}
	if last < len(src) {
		cur = append(cur, text{src[last:]})
	}
	if len(stack) > 0 {
		return nil, fmt.Errorf("[?%s] not closed", names[len(names)-1])
	}
	return cur, nil
}

func parseRange(s string) (int, int, error) {
	a, b, ok := strings.Cut(s, "-")
	if !ok {
		b = a
	}
	lo, err1 := strconv.Atoi(a)
	hi, err2 := strconv.Atoi(b)
	if err1 != nil || err2 != nil || lo > hi || lo < 0 {
		return 0, 0, fmt.Errorf("range %q", s)
	}
	return lo, hi, nil
}

func parsePlaceholder(s string) (node, error) {
	spec, ann, hasAnn := strings.Cut(s, "@")
	spec = strings.TrimSpace(spec)
	head, rest, _ := strings.Cut(spec, "|")
	gen, slot, _ := strings.Cut(head, ":")
	switch gen {
	case "pad":
		lo, hi, err := parseRange(slot)
		return pad{lo, hi}, err
	case "LINES":
		name, r, ok := strings.Cut(slot, ":")
		if !ok {
			return nil, fmt.Errorf("expected LINES:block:N-M")
		}
		lo, hi, err := parseRange(r)
		return lines{name, lo, hi}, err
	case "ONE":
		return one{strings.Split(slot, ",")}, nil
	}
	if !generators[gen] {
		return nil, fmt.Errorf("unknown generator %q", gen)
	}
	if gen == "PER" && !strings.Contains(spec, "form=") {
		return nil, fmt.Errorf("PER needs a form")
	}
	p := placeholder{gen: gen, slot: slot, args: map[string]string{}}
	if rest != "" {
		for _, kv := range strings.Split(rest, "|") {
			k, v, ok := strings.Cut(kv, "=")
			if !ok {
				return nil, fmt.Errorf("argument %q", kv)
			}
			p.args[strings.TrimSpace(k)] = strings.TrimSpace(v)
		}
	}
	if f, ok := p.args["form"]; ok && gen == "PER" && !personForms[f] {
		return nil, fmt.Errorf("unknown PER form %q", f)
	}
	if f, ok := p.args["famille"]; ok && !gtemplate.KnownFamilies[f] {
		return nil, fmt.Errorf("unknown ORG family %q", f)
	}
	if hasAnn {
		parts := strings.Split(strings.TrimSpace(ann), "/")
		if len(parts) < 2 || len(parts) > 3 || !types[parts[0]] || !masks[parts[1]] {
			return nil, fmt.Errorf("annotation %q: expected TYPE/MASK[/SENSITIVE]", ann)
		}
		p.typ, p.annotated = parts[0], true
		if parts[1] != "-" {
			p.mask = parts[1]
		}
		if len(parts) == 3 {
			if !sensitives[parts[2]] {
				return nil, fmt.Errorf("sensitive label %q", parts[2])
			}
			p.sensitive = parts[2]
		}
	}
	return p, nil
}
