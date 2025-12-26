#!/bin/bash

ACTIVATION_LINE='eval "$(mise activate bash)"'
BASHRC="$HOME/.bashrc"

# Check if the line already exists
if grep -Fxq "$ACTIVATION_LINE" "$BASHRC"; then
  echo "Already present in .bashrc"
else
  echo "Adding mise activation line to .bashrc..."
  echo "" >> "$BASHRC"
  echo "# mise active (added by script)" >> "$BASHRC"
  echo "$ACTIVATION_LINE" >> "$BASHRC"
  echo "Successfully added."
fi

echo "Reloading .bashrc"
source "$BASHRC"

# Run diagnostic
echo "Running 'mise doctor' to verify setup"
mise doctor