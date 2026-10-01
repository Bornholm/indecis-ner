// Package llm calls language models through their command-line harnesses,
// Claude Code and Pi, without tools, in an empty temporary directory, as
// indecis' teachers do.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

// Model is a harness and a model.
type Model struct {
	Name string // shown in logs
	args []string
	file bool   // the prompt goes in prompt.md, passed as @prompt.md
	out  string // claude-json or pi-json
}

// Claude runs claude --print with the given model (sonnet, opus...).
func Claude(model string) Model {
	return Model{Name: "claude-" + model, out: "claude-json",
		args: []string{"claude", "--print", "--output-format", "json", "--model", model, "--tools", "", "--strict-mcp-config", "--no-session-persistence"}}
}

// Pi runs pi with the Xolo gateway's model, the one indecis' teachers use.
func Pi(model string) Model {
	return Model{Name: "pi-" + model, out: "pi-json", file: true,
		args: []string{"pi", "--print", "--no-session", "--mode", "json", "--no-tools", "--no-skills", "--no-context-files", "--no-prompt-templates", "--provider", "cadoles-xolo", "--model", model}}
}

// Ask sends a system prompt and a message, and returns the answer.
func (m Model) Ask(ctx context.Context, system, message string, timeout time.Duration) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	dir, err := os.MkdirTemp("", "indecis-ner-llm-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(dir)
	args := append(append([]string(nil), m.args...), "--system-prompt", system)
	if m.file {
		if err := os.WriteFile(dir+"/prompt.md", []byte(message), 0o600); err != nil {
			return "", err
		}
		args = append(args, "@prompt.md")
	}
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	cmd.Dir = dir
	if !m.file {
		cmd.Stdin = strings.NewReader(message)
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("%s: %v: %s", m.Name, err, tail(stderr.String()+stdout.String()))
	}
	return extract(m.out, stdout.Bytes())
}

func extract(format string, out []byte) (string, error) {
	if format == "claude-json" {
		var env struct {
			Result  string `json:"result"`
			IsError bool   `json:"is_error"`
		}
		if err := json.Unmarshal(bytes.TrimSpace(out), &env); err != nil {
			return "", fmt.Errorf("claude envelope: %w", err)
		}
		if env.IsError {
			return "", fmt.Errorf("claude error: %s", tail(env.Result))
		}
		return env.Result, nil
	}
	var last string
	for _, line := range bytes.Split(out, []byte("\n")) {
		var ev struct {
			Type    string `json:"type"`
			Message *struct {
				Role    string `json:"role"`
				Content []struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"content"`
			} `json:"message"`
		}
		if json.Unmarshal(bytes.TrimSpace(line), &ev) != nil {
			continue
		}
		if ev.Type == "tool_execution_start" {
			return "", fmt.Errorf("pi called a tool")
		}
		if ev.Type == "message_end" && ev.Message != nil && ev.Message.Role == "assistant" {
			var text string
			for _, c := range ev.Message.Content {
				if c.Type == "text" {
					text += c.Text
				}
			}
			if text != "" {
				last = text
			}
		}
	}
	if last == "" {
		return "", fmt.Errorf("pi produced no message")
	}
	return last, nil
}

func tail(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 300 {
		s = s[len(s)-300:]
	}
	return s
}
