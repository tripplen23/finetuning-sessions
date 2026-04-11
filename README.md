# Finetuning Sessions

Study materials and exercises for LLM fine-tuning sessions.

## Quick Setup

```bash
# 1. Clone the repo
git clone https://github.com/tripplen23/finetuning-sessions.git
cd finetuning-sessions

# 2. Setup Hugging Face skills for OpenCode
./scripts/setup-opencode-skills.sh
```

The setup script will:
- Clone `huggingface/skills` into `skills/` (if not present)
- Create symlinks in `.opencode/skills/` (local, portable via Git)
- Create symlinks in `~/.config/opencode/skills/` (global)

### Available HF Skills (12 skills)

| Skill | Use for |
|-------|---------|
| `hf-cli` | Hub CLI operations (download, upload, auth, cache) |
| `huggingface-community-evals` | Local model evaluations |
| `huggingface-datasets` | Dataset exploration & API queries |
| `huggingface-gradio` | Build Gradio web UIs |
| `huggingface-llm-trainer` | Train/fine-tune LLMs (SFT, DPO, GRPO) |
| `huggingface-paper-publisher` | Publish papers on HF Hub |
| `huggingface-papers` | Search & read research papers |
| `huggingface-tool-builder` | Build HF API scripts |
| `huggingface-trackio` | Track training experiments |
| `huggingface-vision-trainer` | Train vision models |
| `transformers-js` | ML models in JavaScript |
| `hf-mcp` | HF Hub via MCP server tools |

### On a New Machine

```bash
git clone https://github.com/tripplen23/finetuning-sessions.git
cd finetuning-sessions
./scripts/setup-opencode-skills.sh
```

OpenCode will auto-discover all skills on the next session.

## IDE Support

This project supports skills for **Kiro IDE** and **OpenCode** — because the author loves both. The skills are installed via the setup script and work seamlessly with either IDE.

## Project Structure

```
finetuning-sessions/
├── .opencode/           # OpenCode config (tracked)
│   └── skills/          # Skill symlinks (created by setup)
├── scripts/
│   └── setup-opencode-skills.sh
├── skills/              # huggingface/skills repo (gitignored)
├── week0/               # Week 0 materials
├── week1/               # Week 1 materials
├── week2/               # Week 2 materials
└── week3/               # Week 3 materials
```
