---- MODULE TaskMutant ----
EXTENDS Naturals, TLC

(* Mutant of Task: the second "brain" is back. A candidate may be re-scored by
   a second weight table (Scores2), overwriting the first result — exactly the
   duplication review item 1 removes. NoDoubleScoring and ScoredSingleTable
   must be violated by TLC on this model, otherwise the oracle is vacuous. *)

CONSTANT Candidates, Scores, Scores2

VARIABLE pending, ranked, count

Init ==
    /\ pending = Candidates
    /\ ranked = [c \in Candidates |-> 0]
    /\ count = [c \in Candidates |-> 0]

ScoreOne ==
    \E c \in pending:
        /\ pending' = pending \ {c}
        /\ count' = [count EXCEPT ![c] = @ + 1]
        /\ ranked' = [ranked EXCEPT ![c] = Scores[c]]

ScoreAgain ==
    \E c \in Candidates:
        /\ pending' = pending
        /\ count' = [count EXCEPT ![c] = @ + 1]
        /\ ranked' = [ranked EXCEPT ![c] = Scores2[c]]

Next == ScoreOne \/ ScoreAgain

Spec == Init /\ [][Next]_<<pending, ranked, count>>

TypeOK ==
    /\ pending \subseteq Candidates
    /\ count \in [Candidates -> 0..2]
    /\ ranked \in [Candidates -> 0..100000]

NoDoubleScoring == \A c \in Candidates: count[c] <= 1

ScoredSingleTable == \A c \in Candidates: count[c] = 1 => ranked[c] = Scores[c]

====
