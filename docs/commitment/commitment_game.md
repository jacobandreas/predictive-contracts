# The commitment game: equilibria of split commitment / solution rewards

Setting: the commitment tokens are trained with a consistency reward only, the solution tokens with
a mix of consistency and task reward. Model it as a two-player game between a *committer* (P1) and
a *solver* (P2) over a single binary behavior (hack / no hack).

## Game

- P1 predicts `N` (no hack) or `H` (hack). P2 plays `L` (legitimate attempt) or `K` (hack).
- P1 gets `C+` if the prediction matches P2's action, `C-` otherwise. Write `c = C+ - C- > 0`.
- P2 gets `D+` / `D-` for match / mismatch, plus `T+` if the task is solved and `T-` if not.
  Write `d = D+ - D- >= 0`.
- A legitimate attempt solves the task with probability `p` (problem difficulty); a hack solves it
  (earns the hackable reward) with probability `q` (~1 for `modify_tests`). Expected task reward is
  `T_L = p T+ + (1-p) T-` for `L` and `T_H = q T+ + (1-q) T-` for `K`. The **hack advantage** is
  `Delta = T_H - T_L = (q - p)(T+ - T-)`.

Payoffs (P1, P2), dropping P2's constant `D- + T_L`:

|        | `L`            | `K`                 |
|--------|----------------|---------------------|
| `N`    | `C+`, `d`      | `C-`, `Delta`       |
| `H`    | `C-`, `0`      | `C+`, `d + Delta`   |

P1 plays a pure coordination game: it only cares about matching and is indifferent *which* action
gets matched. P2 plays a coordination game tilted toward `K` by `Delta`. Only two numbers matter:
`d` and `Delta`. The levels `C+, C-, D-, T-` are irrelevant; `c` must be positive but its size
does not change any equilibrium.

## Nash equilibria (simultaneous moves)

- `(N, L)` is a NE iff `Delta <= d` (strict iff `<`): P2 would gain `Delta` by hacking but lose
  the match bonus `d`.
- `(H, K)` is a NE iff `Delta >= -d`.
- `(N, K)` and `(H, L)` are never NE: P1 switches its prediction.
- Mixed NE, iff `|Delta| < d`: P2 hacks with probability exactly 1/2 (to make P1 indifferent) and
  P1 predicts `N` with probability `(1 + Delta/d)/2`.

Three regimes:

| regime | NE | reading |
|---|---|---|
| `Delta > d` | unique: `(H, K)` "honest hacking" | the hack advantage beats the solver's consistency bonus; the committer just learns to say "yes" |
| `|Delta| < d` | `(N, L)`, `(H, K)` both strict, plus the unstable mixed NE | two self-consistent worlds; which one you get is a matter of initialization |
| `Delta < -d` | unique: `(N, L)` | hacking is dominated even before consistency |

## Coarse correlated equilibria

Let `mu` be a distribution over the four profiles with masses `m_NL, m_NK, m_HL, m_HK`. The CCE
conditions (no player gains by committing to a fixed action before seeing its recommendation) are

- P1: `m_NL >= m_NK` and `m_HK >= m_HL` (match probability at least `max(P(L), P(K))`).
- P2 vs always-`L`: `(d + Delta) m_HK >= (d - Delta) m_NK`.
- P2 vs always-`K`: `(d - Delta) m_NL >= (d + Delta) m_HL`.

Consequences:

- `Delta > d`: the last constraint forces `m_NL = m_HL = 0`, then P1's forces `m_NK = 0`. **The
  unique CCE is `(H, K)`.** No correlation device can sustain any legitimate solving.
- `Delta < -d`: symmetric; the unique CCE is `(N, L)`.
- `|Delta| < d`: a polytope containing the convex hull of the three NE and more. Its distinctive
  feature is that **unpredicted hacks `(N, K)` can carry positive mass**, bounded by
  `m_NK <= m_NL` and `m_NK <= ((d + Delta)/(d - Delta)) m_HK`: lying hacks must be rarer than
  honest legitimate attempts and cannot outnumber honest hacks by more than that ratio. At
  `Delta = 0` a third each on `(N,L)`, `(N,K)`, `(H,K)` is a (coarse and ordinary) correlated
  equilibrium. Any problem-conditional strategy pair that plays `(N, L)` on some problems and
  `(H, K)` on others is also a CE, with the problem as the correlating signal.
- At the boundary `Delta = d`, `(N, L)` survives only weakly and `m_NK` may be as large as `m_NL`.

So the CCE and NE sets coincide (a single point) in the two outer regimes and differ only in the
middle band, where the CCE set adds correlated mixtures and a bounded amount of unpredicted
hacking, none of which is a Nash equilibrium.

## Sequential version (the solver sees the commitment)

In the actual transcript P2 observes P1's move, so the relevant concept is subgame-perfect
equilibrium. P2's best response to `N` is `L` iff `Delta <= d`; to `H` it is `K` iff
`Delta >= -d`. Hence:

- `Delta > d`: P2 hacks after either commitment; P1 says `H`. Unique.
- `Delta < -d`: P2 plays `L` after either; P1 says `N`. Unique.
- `|Delta| <= d`: **P2 follows whatever P1 committed to**, and P1 earns `C+` either way, so P1
  is exactly indifferent between the two self-fulfilling commitments. The mixed equilibrium
  disappears; which pure outcome obtains is decided by P1's prior. A model whose prior commitment
  is "no" (the base model) is anchored at `(N, L)` for as long as `Delta <= d`.

## What it says about the experiments

- The solver's consistency weight `d` is the only lever that makes commitments *behavioral*.
  With `d = 0` (task reward only on the solution) P2 ignores the commitment, hacks iff
  `Delta > 0`, and the committer is a pure predictor.
- The committer's reward scale `c` never matters for which equilibrium exists; it only affects
  how fast the committer tracks the solver. This is why constant commitments are stable: in the
  anchored world `(N, L)` the constant "no" is exactly optimal.
- Problem heterogeneity: with `q ~ 1`, `Delta(x) = (1 - p(x)) (T+ - T-)`. The "no" anchor holds on
  problem `x` iff `p(x) >= 1 - d/(T+ - T-)`. With equal weights on task and consistency
  (`d = T+ - T-`, as in the seeded runs) it holds for every solvable problem and is weak only at
  `p(x) = 0`; with a smaller `d` the hardest problems flip to the honest-hacking equilibrium, and a
  committer that can read the problem should then predict "hack" on exactly those, giving the
  problem-conditional correlated equilibrium above. The probe result (hacking not predictable from
  the problem for the RLCR models) is what one expects when the anchor holds almost everywhere.
- The observed RLCR play (100% "no", ~5% hacks) is not itself an equilibrium of this game: it is
  the `(N, L)` anchor plus residual unpredicted hacking, which the middle-band CCE analysis allows
  only in bounded amounts and which a fully best-responding solver would eliminate. Stochastic
  policies, exploration, and problems with `Delta(x)` near `d` account for the residue.
