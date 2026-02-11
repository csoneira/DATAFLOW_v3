# GitHub Copilot “client not supported / API version no longer supported” (VS Code) — Troubleshooting

## Symptom
Copilot requests fail with an error similar to:
- `client not supported: bad request: the specified API version is no longer supported`

This typically indicates that the local Copilot client stack (VS Code and/or Copilot extensions) is outdated relative to the server-side API.

## Quick resolution checklist (recommended order)

### 1) Update Copilot extensions (inside VS Code)
1. Open **Extensions** (`Ctrl+Shift+X`).
2. Update **GitHub Copilot**.
3. Update **GitHub Copilot Chat** (if installed).
4. Apply changes:
   - Command Palette (`Ctrl+Shift+P`) → **Developer: Reload Window**
   - If issues persist, fully close VS Code and reopen.

Alternative (Command Palette):
- **Extensions: Check for Extension Updates**
- **Extensions: Update All Extensions**
- **Developer: Reload Window**

### 2) Update VS Code using your system package manager (Linux)
In many Linux installs, the VS Code UI item **“Check for Updates”** may do nothing because updates are managed externally.

Determine install method:
```bash
which code
code --version
