// Command indecis-anon anonymizes documents as go-anon's anon-doc does,
// with an indecis model in place of the CRF: same formats (DOCX, ODT, PDF
// with OCR, CSV/TSV, plain text), same verification and fail-closed mode,
// same cleaning of hidden surfaces. The model masks personal and
// sensitive data (the mask and sensitive questions, see GUIDELINES.md);
// go-anon's regular expressions still catch identifiers and secrets.
//
//	indecis-anon -model runs/work-fam-p576 -input lettre.docx -output lettre_anon.docx
//	indecis-anon -model runs/work-fam-p576 -bias 6 -strict -input note.txt -output note_anon.txt
//
// Not carried over from anon-doc: the hash strategy and the encrypted
// mapping store, internal to go-anon's commands.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/bornholm/go-anon/pkg/anonymizer"
	pkgcsv "github.com/bornholm/go-anon/pkg/csv"
	"github.com/bornholm/go-anon/pkg/docprocessor"
	pkgdocx "github.com/bornholm/go-anon/pkg/docx"
	"github.com/bornholm/go-anon/pkg/ner"
	"github.com/bornholm/go-anon/pkg/ocr"
	pkgodt "github.com/bornholm/go-anon/pkg/odt"
	pkgpdf "github.com/bornholm/go-anon/pkg/pdf"
	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis-ner/recognizer"
)

var walkers = map[string]func(string) (docprocessor.Walker, error){
	".docx": pkgdocx.NewWalkerFromFile,
	".odt":  pkgodt.NewWalkerFromFile,
	".csv":  pkgcsv.NewWalkerFromFile,
	".tsv":  pkgcsv.NewWalkerFromFile,
	".pdf":  pkgpdf.NewWalkerFromFile,
}

func main() {
	model := flag.String("model", "runs/work-fam-p576", "indecis masking model")
	bias := flag.Float64("bias", 5, "decoding bias towards masking: higher leaks less and masks more (README, Leaks)")
	input := flag.String("input", "", "document to anonymize (required)")
	output := flag.String("output", "", "anonymized document (required)")
	strategy := flag.String("strategy", "tag", `"tag" (typed placeholders, reversible with the mapping) or "redact"`)
	strict := flag.Bool("strict", false, "fail-closed: write nothing if the verification finds a leak")
	multiView := flag.Bool("multi-view", true, "detect on several recompositions of the document (go-anon, more passes)")
	verifyDocument := flag.Bool("verify-document", true, "detect again on the recomposed output (go-anon, one more pass)")
	sanitize := flag.Bool("sanitize", true, "strip metadata, comments and revisions")
	ocrMode := flag.String("ocr", "auto", `OCR of bitmap content: "auto", "on" or "off"`)
	threads := flag.Int("threads", 0, "cores (0: all)")
	mappingOut := flag.String("save-mapping-insecure", "", "write the re-identification mapping IN CLEAR to this JSON file")
	flag.Parse()
	if *input == "" || *output == "" {
		flag.Usage()
		os.Exit(2)
	}

	m, err := indecis.Load(*model, indecis.WithInt8(), indecis.WithThreads(*threads),
		indecis.WithSpanBias("mask", *bias), indecis.WithSpanBias("sensitive", *bias))
	if err != nil {
		log.Fatalf("model: %v", err)
	}
	rec := recognizer.New(m, []string{"mask", "sensitive"},
		ner.RegexEntityFilter(ner.BuiltinRegexPatterns), ner.RegexEntityFilter(ner.SecretPatterns()))
	cfg := anonymizer.Config{Strategy: anonymizer.TagReplace, ConsistentMap: true}
	if *strategy == "redact" {
		cfg.Strategy = anonymizer.Redact
	}
	anon := anonymizer.New(rec, cfg)
	var anonOpts []anonymizer.AnonymizeOption
	if *strict {
		anonOpts = append(anonOpts, anonymizer.WithStrictVerification())
	} else {
		anonOpts = append(anonOpts, anonymizer.WithVerification())
	}

	ext := strings.ToLower(filepath.Ext(*input))
	if ext == ".txt" || ext == ".md" || ext == "" {
		text, err := os.ReadFile(*input)
		if err != nil {
			log.Fatal(err)
		}
		res, err := anon.Anonymize(string(text), anonOpts...)
		if err != nil {
			log.Fatalf("anonymization: %v", err)
		}
		if err := os.WriteFile(*output, []byte(res.Text), 0o644); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Anonymized: %s, %d passages replaced %s\n", *output, len(res.Entities), counts(res.Entities))
		saveMapping(*mappingOut, res.Mapping)
		return
	}

	open, ok := walkers[ext]
	if !ok {
		log.Fatalf("format %q not supported (txt, md, docx, odt, pdf, csv, tsv)", ext)
	}
	var procOpts []docprocessor.Option
	if *multiView {
		procOpts = append(procOpts, docprocessor.WithMultiViewDetection())
	}
	if *verifyDocument {
		if *strict {
			procOpts = append(procOpts, docprocessor.WithStrictDocumentVerification())
		} else {
			procOpts = append(procOpts, docprocessor.WithDocumentVerification())
		}
	}
	proc := docprocessor.New(anon, procOpts...)
	walker, err := open(*input)
	if err != nil {
		log.Fatalf("opening %s: %v", *input, err)
	}
	if err := runOCR(walker, *ocrMode); err != nil {
		log.Fatalf("OCR: %v", err)
	}
	session, report, err := proc.ProcessWithReport(walker, anonOpts...)
	if err != nil {
		log.Fatalf("anonymization: %v", err) // strict: nothing written
	}
	if !report.OK() {
		fmt.Fprintf(os.Stderr, "warning: verification found %d leak(s), %d after recomposition, %d in content that cannot be rewritten; -strict refuses such documents\n",
			len(report.Leaks), len(report.DocumentLeaks), len(report.RegionLeaks))
	}
	if *sanitize || *strict {
		policy := docprocessor.DefaultSanitizePolicy()
		policy.Strict, policy.StripMetadata = *strict, *sanitize
		sr, err := docprocessor.Sanitize(walker, policy)
		if err != nil {
			log.Fatalf("sanitization: %v", err)
		}
		if !sr.OK() {
			fmt.Fprintf(os.Stderr, "warning: surfaces not processed: %v\n", sr.Unprocessed)
		}
	}
	saver, ok := walker.(interface{ SaveTo(string) error })
	if !ok {
		log.Fatalf("this format cannot be saved")
	}
	if err := saver.SaveTo(*output); err != nil {
		log.Fatalf("saving %s: %v", *output, err)
	}
	visual, err := proc.VerifyOutput(walker, *output, session)
	if err != nil {
		log.Fatalf("visual verification: %v", err)
	}
	if len(visual) > 0 {
		if *strict {
			os.Remove(*output)
			log.Fatalf("visual verification: %d passage(s) still readable, document destroyed", len(visual))
		}
		fmt.Fprintf(os.Stderr, "WARNING: visual verification: %d passage(s) still readable in %s\n", len(visual), *output)
	}
	fmt.Printf("Anonymized: %s, %d passages replaced (%d occurrences)\n", *output, len(session.Mapping), report.TotalEntities())
	saveMapping(*mappingOut, session.Mapping)
	session.Close()
}

func counts(ents []ner.Entity) string {
	c := map[string]int{}
	for _, e := range ents {
		c[string(e.Type)]++
	}
	var keys []string
	for k := range c {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var parts []string
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s:%d", k, c[k]))
	}
	return strings.Join(parts, " ")
}

func saveMapping(path string, mapping map[string]string) {
	if path == "" {
		return
	}
	b, _ := json.MarshalIndent(mapping, "", "  ")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		log.Fatal(err)
	}
	fmt.Fprintf(os.Stderr, "WARNING: %s holds the re-identification mapping in clear: protect it and delete it after use\n", path)
}

type ocrCapable interface {
	RunOCR(pkgpdf.OCROptions) error
}

// runOCR is anon-doc's: bitmap content is read with Tesseract when it is
// installed (auto), required with "on".
func runOCR(walker docprocessor.Walker, mode string) error {
	if mode == "off" {
		return nil
	}
	capable, ok := walker.(ocrCapable)
	if !ok {
		if mode == "on" {
			return fmt.Errorf("this format has no bitmap content to read")
		}
		return nil
	}
	err := capable.RunOCR(pkgpdf.OCROptions{
		Engine: ocr.NewTesseractExec(), Rasterizer: pkgpdf.NewPdftoppmRasterizer(),
		Lang: "fra", DPI: pkgpdf.DefaultDPI, VerifyEngine: ocr.NewTesseractExecSparse(),
	})
	if err != nil && mode == "on" {
		return err
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: OCR unavailable (%v): bitmap content not analyzed\n", err)
	}
	return nil
}
