# prompt-triage

A small local classifier that guesses whether a prompt to a coding agent is
*trivial* (something you could do yourself in under two minutes) and learns
from your answer to every guess. No LLM call, no cloud: a sentence-embedding
model runs locally and a logistic regression on top of it is refitted after
each label.

Built as a pair of [Claude Code](https://claude.com/claude-code) hooks, but
the binary is plain stdin/stdout and can be wired to anything.

## Why

Asking a large model to change one config value or run the tests is expensive
and keeps you from knowing your own code base. The hook makes the agent hint
instead of doing when a request is trivial, and the classifier learns *your*
definition of trivial from the labels you give.

## How the loop works

1. You submit a prompt. The `UserPromptSubmit` hook (`triage.sh`) embeds it
   with all-MiniLM-L6-v2, scores it with the current weights, shows
   `triage: TRIVIAL (0.67)` and tells the model to ask you one question before
   doing anything else.
2. The model calls `AskUserQuestion` with the prediction and two options,
   **Trivial** and **Not trivial**. You pick one.
3. The `PostToolUse` hook on `AskUserQuestion` (`triage-answer.sh`) records
   your answer for the pending prompt, refits the classifier on the whole
   dataset and shows `triage: labeled TRIVIAL; 12 rows; accuracy over last 12: 58%`.
4. If you said trivial, that hook also injects the guide text so the model
   points you to the file or gives a hint instead of doing the work. If you
   said not trivial, the model handles the request normally.

The model acts on your label, never on the prediction. The prediction is only
there for you to check; the running accuracy tells you how often it is right.
The first predictions come from random weights, so they are a coin flip, and
the weights stay random until both labels have been seen at least once (a fit
on one class would predict that class for everything).

There is no query selection (the "active" part of active learning): every
prompt is labeled. Once accuracy is high, a natural next step is to only ask
when the probability is close to 0.5.

The question itself is asked by the model, so it depends on the model
following the injected instruction; the recording and the refit do not.

## Install

Requires Rust (edition 2024) and `jq` for the hook script.

```sh
git clone https://github.com/Jibril-Frej/prompt-triage
cd prompt-triage
cargo install --path .
triage setup        # downloads the embedding model (~90 MB) once
```

Data lives in `~/.local/share/prompt-triage/` (or `$PROMPT_TRIAGE_DIR`):

| file | content |
| --- | --- |
| `dataset.jsonl` | one line per labeled prompt: id, timestamp, prompt, embedding, probability at prediction time, label |
| `pending.json` | the last scored prompt waiting for its label |
| `weights.json` | the classifier: 384 weights and a bias |
| `models/` | the embedding model cache |
| `hook.log` | stderr of the hook, for debugging |

## Commands

```
triage setup             download the model, create random initial weights
triage hook              read UserPromptSubmit JSON on stdin, print one JSON line (used by the hook script)
triage label trivial|not label the pending prompt by hand and refit
triage predict "<text>"  score a text without keeping it pending
triage stats             dataset size, class balance, accuracy
```

`triage hook` prints `{"verdict":"TRIVIAL","p":0.67}` (`p` is the probability
of trivial) and nothing for an empty prompt or a slash command. `triage label`
prints one line, `labeled TRIVIAL; 12 rows; accuracy over last 12: 58%`.

## Claude Code setup

Save the two scripts below as `~/.claude/hooks/triage.sh` and
`~/.claude/hooks/triage-answer.sh`, make them executable, then register them
in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          { "type": "command", "command": "~/.claude/hooks/triage.sh", "timeout": 20 }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "AskUserQuestion",
        "hooks": [
          { "type": "command", "command": "~/.claude/hooks/triage-answer.sh", "timeout": 10 }
        ]
      }
    ]
  }
}
```

Open `/hooks` once (or restart) so the new settings are picked up. The scripts
themselves are re-read on every call, so editing them needs no reload.

`triage.sh`, run on every prompt:

```bash
#!/usr/bin/env bash
# UserPromptSubmit hook for prompt-triage (https://github.com/Jibril-Frej/prompt-triage).
# Scores the prompt with the local classifier, shows the prediction, and tells
# the model to let the user confirm the label with AskUserQuestion before doing
# anything. The answer is recorded by triage-answer.sh, a PostToolUse hook on
# AskUserQuestion, which also tells the model how to proceed.

triage=$(command -v triage || echo "$HOME/.cargo/bin/triage")
log="$HOME/.local/share/prompt-triage/hook.log"

# `triage hook` reads the hook JSON from stdin and prints {"verdict":"TRIVIAL","p":0.61}
# (p is the probability of trivial), or nothing for slash commands and empty
# prompts. Errors go to the log file.
out=$("$triage" hook 2>>"$log")
[ -z "$out" ] && exit 0
msg=$(jq -r '"triage: \(.verdict) (\(.p))"' <<<"$out")

context="[triage hook] Prediction for this prompt: $msg.
Before anything else, call AskUserQuestion with exactly one question:
  header: \"triage\"
  question: \"$msg. Is this request trivial?\"
  options: \"Trivial\" and \"Not trivial\"
A hook records the answer and then tells you how to proceed. Do not run any other tool and do not start answering before that."

jq -n --arg msg "$msg" --arg ctx "$context" \
  '{systemMessage: $msg, hookSpecificOutput: {hookEventName: "UserPromptSubmit", additionalContext: $ctx}}'
exit 0
```

`triage-answer.sh`, run after every `AskUserQuestion` call (it ignores
questions whose header is not `triage`):

```bash
#!/usr/bin/env bash
# PostToolUse hook on AskUserQuestion for prompt-triage
# (https://github.com/Jibril-Frej/prompt-triage). When the question is the
# triage one (header "triage"), records the chosen option as the label, refits
# the classifier, and tells the model how to proceed.

triage=$(command -v triage || echo "$HOME/.cargo/bin/triage")
log="$HOME/.local/share/prompt-triage/hook.log"

input=$(cat)
header=$(jq -r '.tool_input.questions[0].header // ""' <<<"$input")
[ "$header" = triage ] || exit 0

# The chosen option: an `answers` object keyed by question text, in the tool
# response (or in the tool input when a PreToolUse hook answered for the user).
answer=$(jq -r '[.tool_response, .tool_input] | map(.answers? | objects | to_entries[0].value) | first // ""' <<<"$input")
case "$answer" in
  "Not trivial") label=not ;;
  "Trivial")     label=trivial ;;
  *) echo "triage-answer: no usable answer ($answer) in: $input" >>"$log"; exit 0 ;;
esac

result=$("$triage" label "$label" 2>>"$log") || exit 0

# What the main model must do when the user says the request is trivial.
# Edit this block freely: it is the user-visible behaviour and tone.
guide=$(cat <<'EOF'
This request looks trivial: something the user can do themselves in under two minutes.
Do NOT do it. Do not edit files and do not run the command. Instead:
- for a code or config change: name the file and say what to change in one or two sentences;
- for a shell command: give a hint (but not the full command) and stop.
Then end the turn. If the user replies that they want you to do it anyway, do it.
EOF
)

if [ "$label" = trivial ]; then
  context="[triage hook] $result. The user says this request is trivial.
$guide"
else
  context="[triage hook] $result. The user says this request is not trivial: handle it normally."
fi
jq -n --arg msg "triage: $result" --arg ctx "$context" \
  '{systemMessage: $msg, hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: $ctx}}'
exit 0
```

The `guide` block is the only user-visible behaviour; edit it to change what
the model does or its tone. If the model ever answers the question in a form
the script does not recognise, the full hook input is appended to `hook.log`.

## The classifier

`src/model.rs` is a logistic regression: probability = sigmoid(w · x + b) on
the 384-dimensional embedding. On every label the 385 parameters are refitted
from zero on the whole dataset with full-batch gradient descent and a small L2
penalty, so the result does not depend on the order labels arrived in. The
refit takes about 6 ms at 10 rows, 24 ms at 100 and 120 ms at 1000 on a
24-core CPU; scoring a prompt, including loading the model, takes about 130 ms.

## License

MIT, see `LICENSE`.
