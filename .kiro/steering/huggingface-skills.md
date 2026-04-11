---
inclusion: manual
---

# Hugging Face Skills

This workspace contains the `huggingface/skills` repository — a collection of AI/ML agent skills following the [Agent Skills](https://agentskills.io) standard.

## Available Skills

Skills live in `skills/skills/*/SKILL.md`. Each skill folder contains a `SKILL.md` with YAML frontmatter (`name`, `description`) and markdown instructions.

| Skill | Path |
|-------|------|
| hf-cli | `skills/skills/hf-cli/` |
| huggingface-community-evals | `skills/skills/huggingface-community-evals/` |
| huggingface-datasets | `skills/skills/huggingface-datasets/` |
| huggingface-gradio | `skills/skills/huggingface-gradio/` |
| huggingface-jobs | `skills/skills/huggingface-jobs/` |
| huggingface-llm-trainer | `skills/skills/huggingface-llm-trainer/` |
| huggingface-paper-publisher | `skills/skills/huggingface-paper-publisher/` |
| huggingface-papers | `skills/skills/huggingface-papers/` |
| huggingface-trackio | `skills/skills/huggingface-trackio/` |
| huggingface-vision-trainer | `skills/skills/huggingface-vision-trainer/` |
| transformers-js | `skills/skills/transformers-js/` |

## MCP Server

The Hugging Face MCP server is configured at `.kiro/settings/mcp.json` with URL `https://huggingface.co/mcp?login`.

## Publish Pipeline

When modifying skills or metadata, regenerate artifacts:

```bash
cd skills && bash scripts/publish.sh
```

To verify artifacts are up to date:

```bash
cd skills && bash scripts/publish.sh --check
```

## Key Files

- `skills/gemini-extension.json` — source of truth for MCP URL
- `skills/scripts/generate_ide_configs.py` — generates Cursor + Kiro configs
- `skills/scripts/generate_agents.py` — generates AGENTS.md and README skills table
- `skills/scripts/publish.sh` — orchestrates full publish pipeline
