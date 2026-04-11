# Kế hoạch Triển khai: Tích hợp Kiro vào Hugging Face Skills

## Tổng quan

Triển khai hỗ trợ Kiro IDE/CLI cho repository `huggingface/skills` bằng cách mở rộng publish pipeline, cập nhật README, và bổ sung hướng dẫn Kiro vào skill `hf-mcp`. Sử dụng Python cho script generation và Bash cho publish pipeline.

## Tasks

- [x] 1. Đổi tên và mở rộng script generate_ide_configs.py
  - [x] 1.1 Đổi tên `scripts/generate_cursor_plugin.py` thành `scripts/generate_ide_configs.py`
    - Rename file giữ nguyên toàn bộ nội dung hiện có
    - _Requirements: 3.4_

  - [x] 1.2 Thêm hằng số `KIRO_MCP_CONFIG` và hàm `build_kiro_mcp_config()` vào `generate_ide_configs.py`
    - Thêm `KIRO_MCP_CONFIG = ROOT / ".kiro" / "settings" / "mcp.json"`
    - Thêm hàm `build_kiro_mcp_config()` tái sử dụng `extract_mcp_from_gemini()` để sinh Kiro MCP config
    - Hàm trả về dict `{"mcpServers": {"<server_name>": {"url": "<url>"}}}`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.4_

  - [x] 1.3 Cập nhật hàm `main()` trong `generate_ide_configs.py` để sinh và kiểm tra `.kiro/settings/mcp.json`
    - Sinh `.kiro/settings/mcp.json` cùng lúc với `.cursor-plugin/plugin.json` và `.mcp.json`
    - Chế độ `--check` kiểm tra cả file Kiro config, báo lỗi nếu lỗi thời
    - In tên file đã ghi khi chạy ở chế độ generate
    - _Requirements: 2.4, 3.1, 3.2_

  - [x] 1.4 Viết property test cho `build_kiro_mcp_config()` — cấu trúc JSON hợp lệ
    - **Property 1: Kiro MCP config có cấu trúc JSON hợp lệ**
    - Dùng hypothesis sinh ngẫu nhiên các biến thể `gemini-extension.json`
    - Assert: output có key `mcpServers`, chứa ít nhất 1 server entry với `url` là string không rỗng
    - **Validates: Requirements 2.2, 2.4**

  - [x] 1.5 Viết property test cho URL consistency giữa Kiro và Cursor configs
    - **Property 2: URL nhất quán giữa Kiro và Cursor configs**
    - Dùng hypothesis sinh ngẫu nhiên các biến thể `gemini-extension.json`
    - Gọi cả `build_kiro_mcp_config()` và `build_mcp_config()`, assert URL bằng nhau
    - **Validates: Requirements 2.3, 3.4**

- [x] 2. Cập nhật publish.sh
  - [x] 2.1 Cập nhật `GENERATED_FILES`, lệnh gọi script, và help text trong `publish.sh`
    - Thêm `.kiro/settings/mcp.json` vào array `GENERATED_FILES`
    - Đổi `generate_cursor_plugin.py` thành `generate_ide_configs.py` trong lệnh `uv run`
    - Cập nhật help text liệt kê `.kiro/settings/mcp.json`
    - _Requirements: 3.1, 3.2, 3.3_

- [x] 3. Checkpoint — Kiểm tra pipeline hoạt động
  - Ensure all tests pass, ask the user if questions arise.
  - Chạy `uv run scripts/generate_ide_configs.py` để xác nhận sinh đúng 3 files
  - Chạy `uv run scripts/generate_ide_configs.py --check` để xác nhận check mode hoạt động

- [x] 4. Cập nhật README.md với hướng dẫn cài đặt Kiro
  - [x] 4.1 Thêm section "### Kiro" vào phần Installation trong `README.md`, sau section Cursor
    - Hướng dẫn copy/symlink skill folders từ `skills/` vào `.kiro/skills/` (workspace) hoặc `~/.kiro/skills/` (global)
    - Hướng dẫn cấu hình MCP server trong `.kiro/settings/mcp.json`
    - Ví dụ JSON code block cho nội dung `.kiro/settings/mcp.json`
    - Đề cập file `.kiro/settings/mcp.json` có sẵn trong repo
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 5. Cập nhật hf-mcp SKILL.md với hướng dẫn Kiro
  - [x] 5.1 Thêm hướng dẫn setup MCP cho Kiro vào `hf-mcp/skills/hf-mcp/SKILL.md`
    - Thêm sau dòng setup link hiện có (`Setup: https://huggingface.co/settings/mcp`)
    - Chỉ rõ đường dẫn `.kiro/settings/mcp.json` và nội dung JSON cần thiết
    - _Requirements: 5.1, 5.2_

- [x] 6. Đảm bảo tương thích định dạng Skills
  - [x] 6.1 Viết property test cho SKILL.md frontmatter validity
    - **Property 3: Mọi skill đều có SKILL.md hợp lệ với frontmatter**
    - Dùng hypothesis sinh ngẫu nhiên SKILL.md content với YAML frontmatter
    - Gọi `parse_frontmatter()`, assert `name` và `description` là string không rỗng khi có mặt
    - **Validates: Requirements 4.1, 4.3**

- [x] 7. Final checkpoint — Xác nhận toàn bộ tích hợp
  - Ensure all tests pass, ask the user if questions arise.
  - Xác nhận `.kiro/settings/mcp.json` được sinh đúng
  - Xác nhận README có section Kiro với nội dung đầy đủ
  - Xác nhận hf-mcp SKILL.md có hướng dẫn Kiro
  - Xác nhận `publish.sh --check` pass

## Ghi chú

- Tasks đánh dấu `*` là optional, có thể bỏ qua để triển khai MVP nhanh hơn
- Mỗi task tham chiếu đến requirements cụ thể để đảm bảo traceability
- Checkpoints đảm bảo kiểm tra tăng dần sau mỗi giai đoạn
- Property tests dùng thư viện `hypothesis` (Python) để kiểm tra correctness properties
- Requirement 4.2 (Kiro nhận diện và load skill) là hành vi runtime của Kiro, không cần code thay đổi
