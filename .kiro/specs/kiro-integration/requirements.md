# Tài liệu Yêu cầu — Tích hợp Kiro vào Hugging Face Skills

## Giới thiệu

Tính năng này bổ sung hỗ trợ Kiro (IDE và CLI) vào repository Hugging Face Skills (`huggingface/skills`). Hiện tại repo đã hỗ trợ Claude Code, OpenAI Codex, Google Gemini CLI và Cursor. Kiro hỗ trợ chuẩn mở Agent Skills (agentskills.io) — cùng định dạng `SKILL.md` với YAML frontmatter mà repo đang sử dụng. Kiro đọc skills từ `.kiro/skills/` (workspace) hoặc `~/.kiro/skills/` (global), và hỗ trợ MCP servers natively. Mục tiêu là đảm bảo người dùng Kiro có thể cài đặt và sử dụng Hugging Face Skills một cách dễ dàng, tương đương với các agent tool khác.

## Thuật ngữ

- **Skills_Repository**: Repository `huggingface/skills` trên GitHub, chứa các Agent Skills cho AI/ML tasks
- **Kiro**: AI assistant và IDE hỗ trợ chuẩn Agent Skills, đọc skills từ `.kiro/skills/` hoặc `~/.kiro/skills/`
- **SKILL_File**: File `SKILL.md` với YAML frontmatter (name, description) chứa hướng dẫn cho AI agent
- **MCP_Server**: Model Context Protocol server, Kiro cấu hình MCP qua `.kiro/settings/mcp.json`
- **Publish_Pipeline**: Script `scripts/publish.sh` tự động sinh metadata cho các agent tool được hỗ trợ
- **README**: File `README.md` gốc của repo, chứa hướng dẫn cài đặt cho từng agent tool
- **Skills_Directory**: Thư mục `skills/` trong repo chứa các skill folders với `SKILL.md`
- **Kiro_Skills_Path**: Đường dẫn `.kiro/skills/` trong workspace hoặc `~/.kiro/skills/` global nơi Kiro tìm skills
- **MCP_Config**: File cấu hình MCP của Kiro tại `.kiro/settings/mcp.json`

## Yêu cầu

### Yêu cầu 1: Hướng dẫn cài đặt Kiro trong README

**User Story:** Là một người dùng Kiro, tôi muốn có hướng dẫn cài đặt Hugging Face Skills cho Kiro trong README, để tôi biết cách sử dụng skills trong Kiro IDE/CLI.

#### Tiêu chí chấp nhận

1. WHEN một người dùng mở file README của Skills_Repository, THE README SHALL hiển thị một section "Kiro" trong phần Installation, ngang hàng với các section Claude Code, Codex, Gemini CLI và Cursor hiện có.
2. THE README SHALL cung cấp hướng dẫn từng bước để copy hoặc symlink skill folders từ thư mục `skills/` của repo vào Kiro_Skills_Path (`.kiro/skills/` trong workspace hoặc `~/.kiro/skills/` global).
3. THE README SHALL cung cấp hướng dẫn cấu hình Hugging Face MCP server trong MCP_Config (`.kiro/settings/mcp.json`) với URL `https://huggingface.co/mcp?login`.
4. THE README SHALL cung cấp ví dụ cụ thể về nội dung file `.kiro/settings/mcp.json` ở dạng JSON code block.
5. IF section Kiro bị thiếu trong README sau khi chạy Publish_Pipeline, THEN THE Publish_Pipeline SHALL báo lỗi validation.

### Yêu cầu 2: Cấu hình MCP Server cho Kiro

**User Story:** Là một người dùng Kiro, tôi muốn có file cấu hình MCP sẵn sàng sử dụng, để tôi có thể kết nối Kiro với Hugging Face MCP server mà không cần cấu hình thủ công.

#### Tiêu chí chấp nhận

1. THE Skills_Repository SHALL chứa một file mẫu `.kiro/settings/mcp.json` với cấu hình Hugging Face MCP server.
2. THE MCP_Config mẫu SHALL sử dụng định dạng JSON hợp lệ với key `mcpServers` chứa entry cho Hugging Face MCP server.
3. THE MCP_Config mẫu SHALL sử dụng URL `https://huggingface.co/mcp?login` cho Hugging Face MCP server, nhất quán với URL trong `.mcp.json` (Cursor) và `gemini-extension.json` (Gemini).
4. WHEN Publish_Pipeline được chạy, THE Publish_Pipeline SHALL sinh hoặc cập nhật file `.kiro/settings/mcp.json` từ cùng nguồn MCP URL đã dùng cho Cursor và Gemini (file `gemini-extension.json`).

### Yêu cầu 3: Tích hợp Publish Pipeline

**User Story:** Là một contributor của repo, tôi muốn Publish_Pipeline tự động sinh metadata cho Kiro, để metadata Kiro luôn đồng bộ với các agent tool khác khi có thay đổi.

#### Tiêu chí chấp nhận

1. WHEN script `publish.sh` được chạy, THE Publish_Pipeline SHALL sinh file `.kiro/settings/mcp.json` cùng với các artifact hiện có (AGENTS.md, README.md, .cursor-plugin/plugin.json, .mcp.json).
2. WHEN script `publish.sh --check` được chạy, THE Publish_Pipeline SHALL kiểm tra file `.kiro/settings/mcp.json` có đồng bộ với nguồn dữ liệu hay không, và báo lỗi nếu file đã lỗi thời.
3. THE Publish_Pipeline SHALL thêm `.kiro/settings/mcp.json` vào danh sách `GENERATED_FILES` trong `publish.sh`.
4. THE Publish_Pipeline SHALL tái sử dụng logic trích xuất MCP URL từ `gemini-extension.json` đã có trong `generate_cursor_plugin.py` để đảm bảo tính nhất quán.

### Yêu cầu 4: Tương thích định dạng Skills

**User Story:** Là một người dùng Kiro, tôi muốn các Hugging Face Skills hoạt động trực tiếp trong Kiro mà không cần chỉnh sửa, để tôi có thể sử dụng ngay sau khi cài đặt.

#### Tiêu chí chấp nhận

1. THE SKILL_File định dạng hiện tại (YAML frontmatter với `name` và `description`, theo sau là nội dung Markdown) SHALL tương thích trực tiếp với Kiro mà không cần chuyển đổi.
2. WHEN một skill folder được copy vào Kiro_Skills_Path, THE Kiro SHALL nhận diện và load skill đó dựa trên file `SKILL.md` trong folder.
3. THE Skills_Repository SHALL duy trì cấu trúc thư mục `skills/<skill-name>/SKILL.md` hiện tại, tương thích với chuẩn Agent Skills mà Kiro hỗ trợ.

### Yêu cầu 5: Hướng dẫn MCP trong Skill hf-mcp

**User Story:** Là một người dùng Kiro muốn sử dụng HF MCP server, tôi muốn skill `hf-mcp` có hướng dẫn cấu hình cho Kiro, để tôi biết cách kết nối MCP server trong Kiro.

#### Tiêu chí chấp nhận

1. THE SKILL_File của `hf-mcp` (tại `hf-mcp/skills/hf-mcp/SKILL.md`) SHALL bao gồm hướng dẫn setup MCP cho Kiro, bên cạnh link setup chung hiện có.
2. THE hướng dẫn setup Kiro trong hf-mcp SKILL_File SHALL chỉ rõ đường dẫn file cấu hình `.kiro/settings/mcp.json` và nội dung JSON cần thiết.
