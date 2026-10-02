// Command bench times a model on a long document built from test
// documents: about -pages A4 pages of 3,000 characters.
//
//	bench -model runs/work-ind -pages 10 -threads 1
//	bench -backbone ~/.cache/indecis/models/bekko-embedding-v1-a25m -pages 10
//
// -backbone times an untrained model with the same questions: the same
// computation, meaningless answers.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"time"

	"github.com/bornholm/indecis"
	"github.com/bornholm/indecis/dataset"
)

func main() {
	model := flag.String("model", "", "trained model directory")
	backbone := flag.String("backbone", "", "backbone directory, timed untrained")
	source := flag.String("source", "data/prose/test.jsonl", "documents joined into the long one")
	pages := flag.Int("pages", 10, "A4 pages of 3,000 characters")
	threads := flag.Int("threads", 1, "cores (0: all)")
	int8 := flag.Bool("int8", true, "int8 inference")
	runs := flag.Int("runs", 3, "timed runs, after a warm-up")
	maxLen := flag.Int("maxlen", 512, "window length (untrained model)")
	flag.Parse()

	ex, err := dataset.ReadFile(*source)
	if err != nil {
		log.Fatal(err)
	}
	var b strings.Builder
	for i := 0; b.Len() < *pages*3000; i++ {
		b.WriteString(ex[i%len(ex)].Text)
		b.WriteString("\n\n")
	}
	text := b.String()

	opts := []indecis.Option{indecis.WithThreads(*threads)}
	if *int8 {
		opts = append(opts, indecis.WithInt8())
	}
	var m *indecis.Model
	if *model != "" {
		m, err = indecis.Load(*model, opts...)
	} else {
		schema := indecis.Schema{
			indecis.NewSpans("entities", "", "PERSON", "CODE", "LOC", "ORG", "DEM", "DATETIME", "QUANTITY", "MISC"),
			indecis.NewSpans("mask", "", "DIRECT", "QUASI"),
			indecis.NewSpans("sensitive", "", "HEALTH", "UNION", "BELIEF", "POLITICS", "ORIGIN", "SEXUALITY", "OFFENCE"),
		}
		m, err = indecis.New(*backbone, schema, 1, append(opts, indecis.WithMaxLen(*maxLen))...)
		if err == nil {
			// Saved then loaded, as a trained model is served: the
			// weights are then read in place instead of in float32.
			dir, derr := os.MkdirTemp("", "bench-model-")
			if derr != nil {
				log.Fatal(derr)
			}
			defer os.RemoveAll(dir)
			if err := m.Save(dir); err != nil {
				log.Fatal(err)
			}
			m = nil
			runtime.GC()
			debug.FreeOSMemory()
			m, err = indecis.Load(dir, opts...)
		}
	}
	if err != nil {
		log.Fatal(err)
	}
	loaded := memory()
	ctx := context.Background()
	if _, err := m.Decide(ctx, text); err != nil { // warm-up: packing, pages
		log.Fatal(err)
	}
	var total time.Duration
	for range *runs {
		start := time.Now()
		if _, err := m.Decide(ctx, text); err != nil {
			log.Fatal(err)
		}
		total += time.Since(start)
	}
	avg := total / time.Duration(*runs)
	fmt.Printf("%d characters, %d words: %.2f s (%.0f ms per page)\n", len([]rune(text)), len(strings.Fields(text)), avg.Seconds(), avg.Seconds()*1000/float64(*pages))
	end := memory()
	fmt.Printf("memory after loading: %s\nmemory after the runs: %s\n", loaded, end)
}

// memory reads the process's memory from /proc: resident (current and
// peak), and how much of it is its own (anonymous) or mapped files (the
// weights, read in place, shared and reclaimable).
func memory() string {
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return "unavailable"
	}
	v := map[string]string{}
	for _, line := range strings.Split(string(b), "\n") {
		if k, val, ok := strings.Cut(line, ":"); ok {
			v[k] = strings.TrimSpace(val)
		}
	}
	mb := func(k string) string {
		var kb int
		fmt.Sscanf(v[k], "%d", &kb)
		return fmt.Sprintf("%d MB", kb/1024)
	}
	return fmt.Sprintf("resident %s (peak %s), own %s, mapped files %s", mb("VmRSS"), mb("VmHWM"), mb("RssAnon"), mb("RssFile"))
}
