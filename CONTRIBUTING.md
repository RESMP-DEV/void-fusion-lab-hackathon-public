# Contributing

1. Start from `origin/master`.
2. Create an ignored worktree under `.worktrees/<slug>/` for multi-file work.
3. Keep one PR to one bounded contribution and name its owned files.
4. Run `just check` and `just quality` in the PR checkout.
5. Record commands, exit codes, elapsed time, scope, and non-claims in the PR.
6. Do not request a human reviewer unless the user explicitly names one.
7. Resolve every actionable review thread with evidence.
8. The representative merges with linear history and verifies `origin/master`.

Evidence claims must use one of these states precisely:

- `implemented`: source exists.
- `exercised`: a real command ran.
- `measured`: a metric was captured.
- `proposed`: no executable path exists yet.
