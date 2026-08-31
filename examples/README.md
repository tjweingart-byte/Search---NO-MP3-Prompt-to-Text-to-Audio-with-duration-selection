# Style examples

Drop briefings in here that you'd want FAM to sound like. They get shown to the
model as examples of the house voice before it writes.

**This is the most direct control you have over the writing.** Describing a
style in rules works loosely; showing three scripts works well. The model learns
the voice, the rhythm and the shape — not the facts, so the topics can be
anything.

## Format

One file per example. Name it `<minutes>-<slug>.txt`, so the model can see how a
briefing scales with length:

```
examples/1-offside-rule.txt
examples/3-interest-rates.txt
examples/5-suez-canal.txt
```

First line is the query someone would have typed, then a blank line, then the
script exactly as it should be spoken:

```
what is the offside rule

A player is offside if they're nearer the opponent's goal than both the ball
and the second-last defender at the moment a teammate plays it forward...
```

## What makes one useful

- **Write it to be heard, not read.** Say it out loud. If you stumble, rewrite.
- **The first sentence matters most.** It's the thing the model copies hardest,
  and the thing a listener judges you on. Make it a fact, not a preamble.
- **Vary the shape.** An explainer, a recap, a "why does this happen" — briefings
  aren't all the same form, and one example teaches one form.
- **Include a short one and a long one.** How a briefing grows from one minute to
  five is exactly where padding creeps in; showing it is better than describing it.
- **End the way you want it to end.** Endings get copied too.
- **Don't over-polish.** Write what you'd genuinely want in your ears on a walk.

Two or three good ones shift the output a lot. Past about five, returns drop off.

Anything in here is loaded automatically at startup — no code change needed. An
empty folder simply means no examples, and the prompt is unchanged.
