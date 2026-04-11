#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_SKILLS_URL="https://github.com/huggingface/skills.git"
SKILLS_SOURCE="$REPO_ROOT/skills"
OPENCODE_LOCAL="$REPO_ROOT/.opencode/skills"
OPENCODE_GLOBAL="$HOME/.config/opencode/skills"

HF_SKILLS=(
    hf-cli
    huggingface-community-evals
    huggingface-datasets
    huggingface-gradio
    huggingface-llm-trainer
    huggingface-paper-publisher
    huggingface-papers
    huggingface-tool-builder
    huggingface-trackio
    huggingface-vision-trainer
    transformers-js
    hf-mcp
)

echo "==> Setting up Hugging Face skills for OpenCode"
echo ""

# 1. Clone huggingface/skills if not present
if [ ! -d "$SKILLS_SOURCE/.git" ]; then
    echo "==> Cloning huggingface/skills into $SKILLS_SOURCE ..."
    git clone "$HF_SKILLS_URL" "$SKILLS_SOURCE"
    echo ""
else
    echo "==> huggingface/skills already present, pulling latest ..."
    git -C "$SKILLS_SOURCE" pull --ff-only 2>/dev/null || true
    echo ""
fi

# 2. Create symlinks in local .opencode/skills/
echo "==> Creating local symlinks (.opencode/skills/) ..."
mkdir -p "$OPENCODE_LOCAL"

for skill in "${HF_SKILLS[@]}"; do
    if [ "$skill" = "hf-mcp" ]; then
        source_path="$SKILLS_SOURCE/hf-mcp/skills/hf-mcp"
    else
        source_path="$SKILLS_SOURCE/skills/$skill"
    fi

    if [ -d "$source_path" ]; then
        ln -sfn "$source_path" "$OPENCODE_LOCAL/$skill"
        echo "  Linked: $skill"
    else
        echo "  SKIP: $skill (source not found at $source_path)"
    fi
done
echo ""

# 3. Create symlinks in global ~/.config/opencode/skills/
echo "==> Creating global symlinks (~/.config/opencode/skills/) ..."
mkdir -p "$OPENCODE_GLOBAL"

for skill in "${HF_SKILLS[@]}"; do
    if [ "$skill" = "hf-mcp" ]; then
        source_path="$SKILLS_SOURCE/hf-mcp/skills/hf-mcp"
    else
        source_path="$SKILLS_SOURCE/skills/$skill"
    fi

    if [ -d "$source_path" ]; then
        ln -sfn "$source_path" "$OPENCODE_GLOBAL/$skill"
        echo "  Linked: $skill"
    fi
done
echo ""

# 4. Summary
linked_count=$(ls -1d "$OPENCODE_LOCAL"/*/ 2>/dev/null | wc -l | tr -d ' ')
echo "==> Done! $linked_count skills linked."
echo ""
echo "    Local:  $OPENCODE_LOCAL/"
echo "    Global: $OPENCODE_GLOBAL/"
echo ""
echo "Skills will be auto-discovered by OpenCode on next session."
