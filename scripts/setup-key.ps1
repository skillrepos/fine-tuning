# Set your Groq API key for this PowerShell window AND every window you open later.
#
# The key is only needed for the OPTIONAL Lab 5 Step 6 comparison. Every other
# lab runs without it.
#
#   .\scripts\setup-key.ps1            # prompts you to paste the key
#   .\scripts\setup-key.ps1 gsk_...    # or pass it as an argument
#
# If Windows says "running scripts is disabled on this system", run this once
# and try again:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#
# Using Git Bash instead of PowerShell? Use:  source scripts/setup-key.sh

param([string]$Key)

if (-not $Key) {
    $Key = Read-Host "Enter your Groq API key"
}
# Strip spaces and any stray line breaks from the paste.
$Key = ($Key -replace '\s', '')

if (-not $Key) {
    Write-Host "Error: no key entered. Nothing was changed."
    return
}
if (-not $Key.StartsWith("gsk_")) {
    Write-Host "Warning: Groq keys normally start with 'gsk_'. Saving it anyway -"
    Write-Host "         if Lab 5 Step 6 reports an authentication error, re-check it."
}

# 1) This window.
$env:GROQ_API_KEY = $Key

# 2) Future windows: a per-user environment variable (no admin rights needed).
[Environment]::SetEnvironmentVariable("GROQ_API_KEY", $Key, "User")

Write-Host "Done! GROQ_API_KEY is set ($($Key.Length) characters) for this window"
Write-Host "and saved as a user environment variable for new windows."
Write-Host ""
Write-Host "Using VS Code's built-in terminal? Quit VS Code completely and reopen it -"
Write-Host "its terminals only see new environment variables after a full restart."
Write-Host ""
Write-Host "Next: python scripts\check_groq.py   (confirms the key works)"
