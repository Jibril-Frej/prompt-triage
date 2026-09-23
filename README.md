# prompt-triage

A small local classifier that guesses whether a prompt to a coding agent is
*trivial* (something you could do yourself in under two minutes) and learns
from your answer to every guess. No LLM call, no cloud: a sentence-embedding
model runs locally and a logistic regression on top of it is refitted after
each label.

Built as a [Claude Code](https://claude.com/claude-code) `UserPromptSubmit`
hook, but the binary is plain stdin/stdout and can be wired to anything.

## Why

Asking a large model to change one config value or run the tests is expensive
and keeps you from knowing your own code base. The hook makes the agent hint
instead of doing when a request is trivial, and the classifier learns *your*
definition of trivial from the labels you give.

## How the loop works

1. You submit a prompt. The hook embeds it with all-MiniLM-L6-v2, scores it with
   the current weights, and **blocks** the prompt with a message such as
   `triage: TRIVIAL (0.67). Resubmit with t: (trivial) or n: (not trivial) in front to label it.`
   Nothing reaches the model.
2. You press up-arrow, add `t:` or `n:` in front, and submit again. The hook
   records your label for the pending prompt, refits the classifier on the
   whole dataset, shows `triage: labeled TRIVIAL; 12 rows; accuracy over last 12: 58%`,
   and lets the prompt through.
3. If you said trivial, the hook also injects the guide text from `triage.sh`
   so the model points you to the file or gives a hint instead of doing the
   work. If you said not trivial, the model handles the request normally.

The model acts on your label, never on the prediction. The prediction is only
there for you to check; the running accuracy tells you how often it is right.
The first predictions come from random weights, so they are a coin flip.

There is no query selection (the "active" part of active learning): every
prompt is labeled. Once accuracy is high, a natural next step is to only block
when the probability is close to 0.5.

A prompt with a `t:` or `n:` prefix that matches nothing pending is labeled
directly, so you can also skip the round trip by typing the label up front.

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

`triage hook` prints `{"kind":"predict","verdict":"TRIVIAL","p":0.67}` for a
plain prompt, `{"kind":"label","label":"TRIVIAL","rows":12,"recent":12,"accuracy":0.58}`
for a prefixed one, and nothing for an empty prompt or a slash command.

## Claude Code setup

Save the script below as `~/.claude/hooks/triage.sh` and make it executable,
then register it in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          { "type": "command", "command": "~/.claude/hooks/triage.sh", "timeout": 10 }
        ]
      }
    ]
  }
}
```

Open `/hooks` once (or restart) so the new settings are picked up. The script
itself is re-read on every prompt, so editing it needs no reload.

```bash
#!/usr/bin/env bash
# UserPromptSubmit hook: deterministic label-then-act loop with the local
# `triage` classifier (https://github.com/Jibril-Frej/prompt-triage).
#
# 1. A plain prompt is scored and BLOCKED; the verdict is shown to the user.
# 2. The user resubmits it with a `t:` (trivial) or `n:` (not) prefix. The
#    hook records the label, refits the model and lets the prompt through.
# 3. If the label is trivial, the guide text below is injected so the model
#    hints instead of doing. No model call is involved in labeling.

triage=$(command -v triage || echo "$HOME/.cargo/bin/triage")
log="$HOME/.local/share/prompt-triage/hook.log"

# `triage hook` reads the hook JSON from stdin and prints one JSON line:
# {"kind":"predict","verdict":"TRIVIAL","p":0.61} or
# {"kind":"label","label":"TRIVIAL","rows":12,"recent":12,"accuracy":0.58}.
# It prints nothing for slash commands and empty prompts; errors go to the log.
out=$("$triage" hook 2>>"$log")
[ -z "$out" ] && exit 0

if [ "$(jq -r .kind <<<"$out")" = predict ]; then
  reason=$(jq -r '"triage: \(.verdict) (\(.p)). Resubmit with t: (trivial) or n: (not trivial) in front to label it."' <<<"$out")
  jq -n --arg r "$reason" '{decision: "block", reason: $r}'
  exit 0
fi

# What the main model must do when the user labeled the request trivial.
# Edit this block freely: it is the user-visible behaviour and tone.
guide=$(cat <<'EOF'
[triage hook] This request looks trivial: something the user can do themselves in under two minutes.
Do NOT do it. Do not edit files and do not run the command. Instead:
- for a code or config change: name the file and say what to change in one or two sentences;
- for a shell command: give a hint (but not the full command) and stop.
Then end the turn. If the user replies that they want you to do it anyway, do it.
EOF
)

msg=$(jq -r '"triage: labeled \(.label); \(.rows) rows; accuracy over last \(.recent): \((.accuracy * 100) | round)%"' <<<"$out")
context="[triage hook] The leading t: or n: in the prompt is a label marker for the triage hook; ignore it."
if [ "$(jq -r .label <<<"$out")" = TRIVIAL ]; then
  context="$context
$guide"
fi
jq -n --arg msg "$msg" --arg ctx "$context" \
  '{systemMessage: $msg, hookSpecificOutput: {hookEventName: "UserPromptSubmit", additionalContext: $ctx}}'
exit 0
```

The `guide` block is the only user-visible behaviour; edit it to change what
the model does or its tone.

## The classifier

`src/model.rs` is a logistic regression: probability = sigmoid(w · x + b) on
the 384-dimensional embedding. On every label the 385 parameters are refitted
from zero on the whole dataset with full-batch gradient descent and a small L2
penalty, so the result does not depend on the order labels arrived in. The
refit takes about 6 ms at 10 rows, 24 ms at 100 and 120 ms at 1000 on a
24-core CPU; scoring a prompt, including loading the model, takes about 130 ms.

## License

MIT, see `LICENSE`.
