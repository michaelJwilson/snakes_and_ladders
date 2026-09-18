The reinforcement-learning tree figure's caption renders "on a landscape"
again. `qa/rl_tree_policy.py` had come to say "on a environment", which is
ungrammatical and disagreed with the caption committed beside the figure, so
the tree and its own renderer said different things -- the state
`qa/CLAUDE.md`'s "a figure is rendered, never predicted" exists to prevent.
Restoring the word the committed caption carries fixes both in one line and
regenerates nothing.
