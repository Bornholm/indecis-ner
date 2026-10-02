// Command prose has a language model write French workplace documents in
// prose, their personal data labeled inline with markers, and a second
// model check the labels.
//
//	prose -split train -n 400 -out data/prose
//	prose -split test -n 100 -out data/prose
//	prose -split ind-test -n 50 -instructions "..." -forbid data/lexicon/words.txt
//
// Each document gets a scenario of synth/scenarios.tsv (train and test
// scenarios differ), and a cast drawn by go-anon's generators: people,
// organisations, places, so that names do not repeat from one document to
// the next. Claude writes the document with markers
// ⟦TYPE/MASK[/SENSITIVE]⟧…⟦/⟧ following GUIDELINES.md; Pi then lists the
// labels it would change and the passages it finds missing.
//
// Outputs, in -out: <split>.jsonl (every document), <split>_agreed.jsonl
// (documents where Pi changes no masking decision and misses nothing to
// mask) and <split>_review.jsonl (Pi's remarks). Calls are cached in
// -cache: a run resumes where it stopped.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/bornholm/go-anon/pkg/synth/gazetteer"
	"github.com/bornholm/go-anon/pkg/synth/value"
	"github.com/bornholm/indecis-ner/llm"
	"github.com/bornholm/indecis/dataset"
)

const writerPrompt = `Tu écris des documents professionnels français réalistes et fictifs, du monde du travail, dans lesquels tu annotes toi-même les données personnelles.

Annotation : entoure chaque passage de données personnelles par des balises ⟦TYPE/MASQUE⟧ et ⟦/⟧, ou ⟦TYPE/MASQUE/SENSIBLE⟧ et ⟦/⟧.
- TYPE : PERSON (noms, prénoms, initiales, avec civilité ou titre), CODE (numéros : sécurité sociale, matricule, téléphone, courriel, IBAN, référence de dossier), LOC (adresses, villes, sites, bâtiments), ORG (employeurs, services nommés, écoles, hôpitaux, syndicats), DEM (attributs d'une personne : métier, grade, diplôme, âge, nationalité, situation familiale, description physique, diagnostic), DATETIME (dates, heures, durées), QUANTITY (salaire, montants, pourcentages qui concernent une personne), MISC (toute autre information qui décrit une personne : véhicule, loisir, événement vécu).
- MASQUE : DIRECT si le passage identifie seul une personne (nom complet, numéro de sécurité sociale, matricule, courriel ou téléphone personnel, IBAN, adresse personnelle complète) ; QUASI s'il identifie une personne en le combinant avec d'autres informations (prénom ou nom seul, date de naissance, métier, service, site, salaire, date d'embauche, ville, employeur, diagnostic, âge, lien familial) ; NON s'il ne dit rien d'une personne.
- SENSIBLE (facultatif, catégories des articles 9 et 10 du RGPD) : HEALTH (santé, handicap, soins, traitement, motif d'arrêt), UNION (appartenance ou mandat syndical), BELIEF (religion, convictions), POLITICS (opinions ou engagement politique), ORIGIN (origine ethnique), SEXUALITY (vie sexuelle, orientation), OFFENCE (infractions, condamnations).
- Toute personne physique est protégée. Les organismes publics agissant comme tels (URSSAF, CPAM) et les mentions légales ne sont pas annotés.
- Passages minimaux : sans article ni ponctuation autour (« une ⟦DEM/QUASI/HEALTH⟧lombalgie chronique⟦/⟧ »). Jamais de balises imbriquées ni chevauchantes. Un même passage n'a qu'une balise.
- Les périphrases qui désignent une personne sont des quasi-identifiants à annoter : « ⟦DEM/QUASI⟧la seule infirmière de nuit du service⟦/⟧ ».

Écris un document crédible, avec des détails concrets et des quasi-identifiants dans le texte courant, pas seulement dans des en-têtes. Utilise les personnes et lieux fournis, et invente le reste. Réponds uniquement par le document annoté, sans titre de réponse ni commentaire.`

const checkerPrompt = `Tu relis l'annotation des données personnelles d'un document professionnel français. Chaque passage annoté est entouré de ⟦n:TYPE/MASQUE[/SENSIBLE]⟧ et ⟦/n⟧, n étant son numéro.

Les règles d'annotation :
- TYPE : PERSON, CODE, LOC, ORG, DEM (métier, âge, diplôme, situation familiale, diagnostic…), DATETIME, QUANTITY, MISC.
- MASQUE : DIRECT (identifie seul une personne : nom complet, numéro de sécurité sociale, matricule, courriel ou téléphone personnel, IBAN, adresse personnelle complète), QUASI (identifie en combinaison : prénom ou nom seul, date de naissance, métier, service, salaire, ville, employeur, diagnostic, âge, lien familial, périphrase désignant une personne), NON (ne dit rien d'une personne).
- SENSIBLE : HEALTH, UNION, BELIEF, POLITICS, ORIGIN, SEXUALITY, OFFENCE (articles 9 et 10 du RGPD).
- Toute personne physique est protégée.

Signale seulement les erreurs nettes : un masque ou une catégorie sensible faux, un passage de données personnelles oublié. Ne signale pas les questions de bornes ou de type quand le masque est juste.

Réponds uniquement en JSON :
{"corrections": [{"id": 3, "label": "DEM/QUASI/HEALTH"}], "oublis": [{"texte": "texte exact du document", "label": "DEM/QUASI"}]}
avec "NON" comme masque pour un passage qui ne devait pas être annoté. Listes vides si tout est juste.`

var (
	markerRe = regexp.MustCompile(`⟦([A-Z]+)/([A-Z]+)(?:/([A-Z]+))?⟧|⟦/⟧`)
	types    = map[string]bool{"PERSON": true, "CODE": true, "LOC": true, "ORG": true, "DEM": true, "DATETIME": true, "QUANTITY": true, "MISC": true}
	masks    = map[string]bool{"DIRECT": true, "QUASI": true, "NON": true}
	sens     = map[string]bool{"HEALTH": true, "UNION": true, "BELIEF": true, "POLITICS": true, "ORIGIN": true, "SEXUALITY": true, "OFFENCE": true}
)

type span struct {
	Start, End       int
	Type, Mask, Sens string
}

type scenario struct{ split, genre, situation string }

type result struct {
	ID       string            `json:"id"`
	Scenario string            `json:"scenario"`
	Genre    string            `json:"genre"`
	Text     string            `json:"text"`
	Spans    []span            `json:"spans"`
	Check    *check            `json:"check,omitempty"`
	Meta     map[string]string `json:"meta,omitempty"`
}

type check struct {
	Corrections []struct {
		ID    int    `json:"id"`
		Label string `json:"label"`
	} `json:"corrections"`
	Missed []struct {
		Text  string `json:"texte"`
		Label string `json:"label"`
	} `json:"oublis"`
}

func main() {
	split := flag.String("split", "train", "scenarios: train or test")
	n := flag.Int("n", 10, "documents")
	seed := flag.Int64("seed", 1, "seed")
	out := flag.String("out", "data/prose", "output directory")
	cache := flag.String("cache", "data/prose/cache", "call cache")
	writer := flag.String("writer", "sonnet", "Claude model that writes")
	checker := flag.String("checker", "cadoles/minimax-m3", "Pi model that checks (empty: no check)")
	concurrency := flag.Int("concurrency", 2, "documents in progress at once")
	interval := flag.Duration("interval", 5*time.Second, "minimum time between two calls of a model")
	extra := flag.String("instructions", "", "instruction added to every brief")
	forbid := flag.String("forbid", "", "file of words the documents must not use, one per line")
	flag.Parse()
	var forbidden []string
	if *forbid != "" {
		b, err := os.ReadFile(*forbid)
		if err != nil {
			log.Fatal(err)
		}
		for _, w := range strings.Split(string(b), "\n") {
			if w = strings.TrimSpace(w); w != "" {
				forbidden = append(forbidden, w)
			}
		}
	}

	scenarios, err := readScenarios("synth/scenarios.tsv", *split)
	if err != nil {
		log.Fatal(err)
	}
	bundle, err := gazetteer.LoadSeed("fr", gazetteer.DefaultOptions())
	if err != nil {
		log.Fatal(err)
	}
	for _, d := range []string{*out, *cache} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			log.Fatal(err)
		}
	}
	w, c := llm.Claude(*writer), llm.Pi(*checker)
	wTick, cTick := time.NewTicker(*interval), time.NewTicker(*interval)

	results := make([]*result, *n)
	var wg sync.WaitGroup
	sem := make(chan struct{}, *concurrency)
	for i := 0; i < *n; i++ {
		rng := rand.New(rand.NewSource(*seed*1_000_003 + int64(i)))
		sc := scenarios[i%len(scenarios)]
		prompt := brief(sc, rng, bundle)
		if *extra != "" {
			prompt += "\nConsigne : " + *extra + "\n"
		}
		if len(forbidden) > 0 {
			prompt += "\nN'emploie aucun de ces mots ni aucun mot de la même famille, pour les données sensibles : " + strings.Join(forbidden, ", ") + ".\n"
		}
		id := fmt.Sprintf("%s-%04d", *split, i)
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			r, err := produce(id, sc, prompt, w, c, *checker != "", wTick, cTick, *cache)
			if err != nil {
				log.Printf("%s: %v", id, err)
				return
			}
			results[i] = r
			status := "unchecked"
			if r.Check != nil {
				status = fmt.Sprintf("%d corrections, %d missed", len(r.Check.Corrections), len(r.Check.Missed))
			}
			log.Printf("%s: %s, %d passages, %s", id, sc.genre, len(r.Spans), status)
		}()
	}
	wg.Wait()
	if err := write(*out, *split, results); err != nil {
		log.Fatal(err)
	}
}

func readScenarios(path, split string) ([]scenario, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []scenario
	for _, line := range strings.Split(string(b), "\n") {
		f := strings.Split(line, "\t")
		if len(f) == 3 && f[0] == split {
			out = append(out, scenario{f[0], f[1], f[2]})
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no %s scenario", split)
	}
	return out, nil
}

var tones = []string{"formel", "neutre et factuel", "familier entre collègues", "administratif", "inquiet", "agacé mais poli"}

// brief writes the request for one document: the scenario and a cast.
func brief(sc scenario, rng *rand.Rand, bundle *gazetteer.Bundle) string {
	g := value.New(rng, bundle)
	var b strings.Builder
	fmt.Fprintf(&b, "Genre : %s\nSituation : %s\nTon : %s\nLongueur : %d à %d mots.\n\nPersonnes à utiliser (et d'autres si besoin) :\n",
		sc.genre, sc.situation, tones[rng.Intn(len(tones))], 150+50*rng.Intn(4), 300+50*rng.Intn(5))
	for range 2 + rng.Intn(3) {
		p := g.NewPerson()
		sex := "homme"
		if p.Female {
			sex = "femme"
		}
		fmt.Fprintf(&b, "- %s %s (%s)\n", p.First, p.FullLast(), sex)
	}
	org := g.NewOrg()
	a1, a2 := g.NewAddress(), g.NewAddress()
	fmt.Fprintf(&b, "\nOrganisation : %s\nAdresses : %s ; %s\n", org.Denomination, a1.Street()+" "+a1.CityCode(), a2.Street()+" "+a2.CityCode())
	fmt.Fprintf(&b, "Date du document : %02d/%02d/%d\n", 1+rng.Intn(28), 1+rng.Intn(12), 2023+rng.Intn(3))
	return b.String()
}

// produce writes then checks one document, through the cache.
func produce(id string, sc scenario, prompt string, w, c llm.Model, doCheck bool, wTick, cTick *time.Ticker, cache string) (*result, error) {
	ctx := context.Background()
	resp, err := cached(cache, "write", w.Name+prompt, func() (string, error) {
		<-wTick.C
		return w.Ask(ctx, writerPrompt, prompt, 10*time.Minute)
	})
	if err != nil {
		return nil, err
	}
	text, spans, err := parse(resp)
	if err != nil {
		drop(cache, "write", w.Name+prompt)
		return nil, fmt.Errorf("markers: %w", err)
	}
	r := &result{ID: id, Scenario: sc.situation, Genre: sc.genre, Text: text, Spans: spans}
	if !doCheck {
		return r, nil
	}
	numbered := number(text, spans)
	answer, err := cached(cache, "check", c.Name+numbered, func() (string, error) {
		<-cTick.C
		return c.Ask(ctx, checkerPrompt, numbered, 10*time.Minute)
	})
	if err != nil {
		return r, nil // kept unchecked
	}
	var ck check
	raw := answer
	if i, j := strings.Index(raw, "{"), strings.LastIndex(raw, "}"); i >= 0 && j > i {
		raw = raw[i : j+1]
	}
	if err := json.Unmarshal([]byte(raw), &ck); err != nil {
		drop(cache, "check", c.Name+numbered)
		return r, nil
	}
	r.Check = &ck
	return r, nil
}

func key(kind, s string) string {
	h := sha256.Sum256([]byte(s))
	return kind + "-" + hex.EncodeToString(h[:8]) + ".txt"
}

func cached(dir, kind, k string, call func() (string, error)) (string, error) {
	path := filepath.Join(dir, key(kind, k))
	if b, err := os.ReadFile(path); err == nil {
		return string(b), nil
	}
	s, err := call()
	if err != nil {
		return "", err
	}
	return s, os.WriteFile(path, []byte(s), 0o644)
}

func drop(dir, kind, k string) { os.Remove(filepath.Join(dir, key(kind, k))) }

// parse removes the markers and returns the text and its passages, in
// bytes. NON passages are kept as typed but not masked.
func parse(resp string) (string, []span, error) {
	resp = strings.TrimSpace(resp)
	var b strings.Builder
	var spans []span
	open := -1
	var cur span
	last := 0
	for _, m := range markerRe.FindAllStringSubmatchIndex(resp, -1) {
		b.WriteString(resp[last:m[0]])
		last = m[1]
		if m[2] < 0 { // closing
			if open < 0 {
				return "", nil, fmt.Errorf("⟦/⟧ without an opening marker")
			}
			cur.End = b.Len()
			s := strings.TrimSpace(b.String()[cur.Start:cur.End])
			if s == "" {
				return "", nil, fmt.Errorf("empty passage")
			}
			// trim spaces inside the markers
			for cur.Start < cur.End && b.String()[cur.Start] == ' ' {
				cur.Start++
			}
			for cur.End > cur.Start && b.String()[cur.End-1] == ' ' {
				cur.End--
			}
			spans = append(spans, cur)
			open = -1
			continue
		}
		if open >= 0 {
			return "", nil, fmt.Errorf("nested markers")
		}
		t, mk := resp[m[2]:m[3]], resp[m[4]:m[5]]
		s := ""
		if m[6] >= 0 {
			s = resp[m[6]:m[7]]
		}
		if !types[t] || !masks[mk] || (s != "" && !sens[s]) {
			return "", nil, fmt.Errorf("label %s/%s/%s", t, mk, s)
		}
		if mk == "NON" {
			mk = ""
		}
		cur = span{Start: b.Len(), Type: t, Mask: mk, Sens: s}
		open = 1
	}
	b.WriteString(resp[last:])
	if open >= 0 {
		return "", nil, fmt.Errorf("marker not closed")
	}
	if strings.ContainsAny(b.String(), "⟦⟧") {
		return "", nil, fmt.Errorf("malformed marker left in the text")
	}
	return b.String(), spans, nil
}

// number writes the text with numbered markers, for the checker.
func number(text string, spans []span) string {
	var b strings.Builder
	last := 0
	for i, s := range spans {
		b.WriteString(text[last:s.Start])
		mask := s.Mask
		if mask == "" {
			mask = "NON"
		}
		label := s.Type + "/" + mask
		if s.Sens != "" {
			label += "/" + s.Sens
		}
		fmt.Fprintf(&b, "⟦%d:%s⟧%s⟦/%d⟧", i, label, text[s.Start:s.End], i)
		last = s.End
	}
	b.WriteString(text[last:])
	return b.String()
}

// agreed reports whether the checker changes no masking decision and
// finds nothing missing to mask.
func agreed(r *result) bool {
	if r.Check == nil {
		return false
	}
	for _, c := range r.Check.Corrections {
		if c.ID < 0 || c.ID >= len(r.Spans) {
			continue
		}
		parts := strings.Split(c.Label, "/")
		if len(parts) < 2 {
			continue
		}
		want := parts[1]
		if want == "NON" {
			want = ""
		}
		got := r.Spans[c.ID].Mask
		if want != got && !(want == "DIRECT" && got == "QUASI" || want == "QUASI" && got == "DIRECT") {
			return false // masked or not: the decision that matters
		}
	}
	for _, m := range r.Check.Missed {
		if p := strings.Split(m.Label, "/"); len(p) >= 2 && p[1] != "NON" && strings.Contains(r.Text, m.Text) {
			return false
		}
	}
	return true
}

func write(dir, split string, results []*result) error {
	var all, ok []dataset.Example
	var review []*result
	for _, r := range results {
		if r == nil {
			continue
		}
		ents, mask, sensitive := []any{}, []any{}, []any{}
		for _, s := range r.Spans {
			ents = append(ents, map[string]any{"start": s.Start, "end": s.End, "type": s.Type})
			if s.Mask != "" {
				mask = append(mask, map[string]any{"start": s.Start, "end": s.End, "type": s.Mask})
			}
			if s.Sens != "" {
				sensitive = append(sensitive, map[string]any{"start": s.Start, "end": s.End, "type": s.Sens})
			}
		}
		e := dataset.Example{Text: r.Text, Labels: map[string]any{"entities": ents, "mask": mask, "sensitive": sensitive},
			Family: r.Genre + ": " + r.Scenario, Meta: map[string]string{"id": r.ID}}
		all = append(all, e)
		if agreed(r) {
			ok = append(ok, e)
		} else if r.Check != nil {
			review = append(review, r)
		}
	}
	for name, ex := range map[string][]dataset.Example{split: all, split + "_agreed": ok} {
		f, err := os.Create(filepath.Join(dir, name+".jsonl"))
		if err != nil {
			return err
		}
		if err := dataset.WriteJSONL(f, ex); err != nil {
			return err
		}
		f.Close()
	}
	sort.Slice(review, func(a, b int) bool { return review[a].ID < review[b].ID })
	f, err := os.Create(filepath.Join(dir, split+"_review.jsonl"))
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetEscapeHTML(false)
	for _, r := range review {
		enc.Encode(r)
	}
	f.Close()
	log.Printf("%s: %d documents, %d agreed by the checker, %d to review", split, len(all), len(ok), len(review))
	return nil
}
