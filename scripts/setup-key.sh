#!/usr/bin/env bash
# Set your Groq API key for this terminal AND every terminal you open later.
#
# The key is only needed for the OPTIONAL comparison in Lab 5's STEP 5. Every other
# lab runs without it.
#
#   source scripts/setup-key.sh            # prompts you to paste the key
#   source scripts/setup-key.sh gsk_...    # or pass it as an argument
#
# Use "source", not "bash": a script run with bash cannot change the terminal
# you typed it in. (If you do use bash, new terminals still get the key.)
#
# Works in: GitHub Codespaces, macOS Terminal (zsh or bash), Linux, and
# Git Bash on Windows. In Windows PowerShell use scripts\setup-key.ps1 instead.

# --- were we sourced, or run as a separate program? ------------------------
_gk_sourced=0
if [ -n "${ZSH_VERSION:-}" ]; then
  case "${ZSH_EVAL_CONTEXT:-}" in *:file*) _gk_sourced=1 ;; esac
elif [ -n "${BASH_VERSION:-}" ]; then
  [ "${BASH_SOURCE[0]}" != "$0" ] && _gk_sourced=1
fi

_gk_main() {
  local key="${1:-}"
  if [ -z "$key" ]; then
    printf "Enter your Groq API key: "
    read -r key
  fi
  # Strip spaces, tabs, and the stray carriage return a Windows paste can add.
  key=$(printf '%s' "$key" | tr -d '[:space:]')

  if [ -z "$key" ]; then
    echo "Error: no key entered. Nothing was changed."
    return 1
  fi
  case "$key" in
    gsk_*) ;;
    *) echo "Warning: Groq keys normally start with 'gsk_'. Saving it anyway -"
       echo "         if Lab 5's STEP 5 reports an authentication error, re-check it." ;;
  esac

  # 1) This terminal.
  export GROQ_API_KEY="$key"

  # 2) Future terminals: the start-up file for the shell running this script.
  local rc
  if [ -n "${ZSH_VERSION:-}" ]; then
    rc="$HOME/.zshrc"
  elif [ "$(uname -s)" = "Darwin" ]; then
    rc="$HOME/.bash_profile"      # macOS bash reads this, not .bashrc
  else
    rc="$HOME/.bashrc"            # Codespaces, Linux, Git Bash on Windows
  fi
  touch "$rc"
  # Replace any earlier key (re-running this script rotates the key), keeping
  # every other line. No "sed -i": its syntax differs between macOS and Linux.
  local tmp="$rc.groqtmp.$$"
  grep -v '^export GROQ_API_KEY=' "$rc" > "$tmp" || true
  printf "export GROQ_API_KEY='%s'\n" "$key" >> "$tmp"
  cat "$tmp" > "$rc" && rm -f "$tmp"

  echo "Done! GROQ_API_KEY is set (${#key} characters) and saved in $rc"
  echo "Next: python scripts/check_groq.py   (confirms the key works)"
}

_gk_main "$@"
_gk_status=$?

if [ "$_gk_sourced" = "0" ] && [ "$_gk_status" = "0" ]; then
  echo
  echo "Note: you ran this without 'source', so THIS terminal does not have the key yet."
  echo "      Open a new terminal, or run:  source scripts/setup-key.sh"
fi

unset -f _gk_main
unset _gk_sourced
if [ "$_gk_status" != "0" ]; then
  unset _gk_status
  # 'return' when sourced (so the terminal stays open), 'exit' otherwise.
  return 1 2>/dev/null || exit 1
fi
unset _gk_status
