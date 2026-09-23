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

1. You submit a prompt. The hook embeds it with all-MiniLM-L6-v2, scores it
   with the current weights, keeps it pending, and **blocks** it with the
   message `triage: TRIVIAL (0.67). Reply t (trivial) or n (not trivial).`
   Claude Code shows your prompt under that message. Nothing has been sent to
   the model yet.
2. You reply with a single `t` or `n` (or `trivial` / `not`, any case). The
   hook records your label for the pending prompt, refits the classifier on
   the whole dataset and shows
   `triage: labeled TRIVIAL; 12 rows; accuracy over last 12: 58%`.
3. Your reply goes to the model with the original prompt injected as context,
   so the model answers the original prompt. If you said trivial, the hook
   also injects the guide text so the model points you to the file or gives a
   hint instead of doing the work. If you said not trivial, the model handles
   the request normally.

The only model call is the one that answers your request, so the prediction
appears about 150 ms after you press enter.

The model acts on your label, never on the prediction. The prediction is only
there for you to check; the running accuracy tells you how often it is right.
The first predictions come from random weights, so they are a coin flip, and
the weights stay random until both labels have been seen at least once (a fit
on one class would predict that class for everything).

Prompts longer than 1500 characters are reported as not trivial without
scoring and go straight through, unlabeled. The embedding model only reads
the first 512 tokens (about 2000 characters), so a longer prompt would be
judged on its beginning alone, and a long request is not a two-minute task
anyway.

A `t` or `n` with nothing pending is passed through as an ordinary prompt.
Sending a different prompt instead of a label drops the pending one.

There is no query selection (the "active" part of active learning): every
prompt is labeled. Once accuracy is high, a natural next step is to only
block when the probability is close to 0.5.

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
prompt (`p` is the probability of trivial; `"long":true` is added when the
prompt was too long to score),
`{"kind":"label","label":"TRIVIAL","rows":12,"recent":12,"accuracy":0.58,"prompt":"..."}`
for a label reply, and nothing for an empty prompt, a slash command, or a
label reply with nothing pending.

## Claude Code setup

Save the script below as `~/.claude/hooks/triage.sh` and make it executable,
then register it in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          { "type": "command", "command": "~/.claude/hooks/triage.sh", "timeout": 20 }
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
# UserPromptSubmit hook for prompt-triage (https://github.com/Jibril-Frej/prompt-triage).
#
# 1. A prompt is scored and BLOCKED; the prediction is shown to the user and
#    nothing reaches the model.
# 2. The user replies with a single `t` (trivial) or `n` (not trivial). The
#    hook records the label, refits the classifier, and lets that reply through
#    with the original prompt injected as context, so the model answers it.
# 3. If the label is trivial, the guide text below is injected too, so the
#    model hints instead of doing. No model call happens before the label.

triage=$(command -v triage || echo "$HOME/.cargo/bin/triage")
log="$HOME/.local/share/prompt-triage/hook.log"

# `triage hook` reads the hook JSON from stdin and prints one JSON line:
# {"kind":"predict","verdict":"TRIVIAL","p":0.61}, with "long":true when the
# prompt was too long to score, or
# {"kind":"label","label":"TRIVIAL","rows":12,"recent":12,"accuracy":0.58,"prompt":"..."}.
# It prints nothing for slash commands, empty prompts, and a label reply with
# nothing pending; errors go to the log file.
out=$("$triage" hook 2>>"$log")
[ -z "$out" ] && exit 0

if [ "$(jq -r .kind <<<"$out")" = predict ]; then
  if [ "$(jq -r '.long // false' <<<"$out")" = true ]; then
    jq -n '{systemMessage: "triage: NOT (long prompt, not scored)"}'
    exit 0
  fi
  reason=$(jq -r '"triage: \(.verdict) (\(.p)). Reply t (trivial) or n (not trivial)."' <<<"$out")
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
If the request is trivial, your tone should be passive aggressive, making the user understand that waht they ask is expensive and will make them dumber.
Then end the turn. If the user replies that they want you to do it anyway, do it.
EOF
)

msg=$(jq -r '"triage: labeled \(.label); \(.rows) rows (\(.trivial_rows) trivial, \(.rows - .trivial_rows) not); accuracy over last \(.recent): \((.accuracy * 100) | round)%"' <<<"$out")prompt=$(jq -r .prompt <<<"$out")
context="[triage hook] The user's message above is only a label reply for the triage hook. Their actual request is the following; answer it:
$prompt"
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

One consequence of this flow: in the transcript your message is the letter,
and the real prompt lives in the hook context attached to it. Claude Code's
prompt history (up-arrow) still has the full prompt.

## The classifier

`src/model.rs` is a logistic regression: probability = sigmoid(w · x + b) on
the 384-dimensional embedding. On every label the 385 parameters are refitted
from zero on the whole dataset with full-batch gradient descent and a small L2
penalty, so the result does not depend on the order labels arrived in. The
refit takes about 6 ms at 10 rows, 24 ms at 100 and 120 ms at 1000 on a
24-core CPU; scoring a prompt, including loading the model, takes about 130 ms.

## Update

If you make a change to the codebase, remember to rebuild and reinstall:

```
cargo install --path . 
```


## License

MIT, see `LICENSE`.
