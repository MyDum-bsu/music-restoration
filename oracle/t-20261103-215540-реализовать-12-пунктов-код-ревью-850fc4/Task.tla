---- MODULE Task ----
EXTENDS Naturals, TLC

(* Review item 1 in miniature: ONE scorer, ONE weight table.
   Scores is the single table (it lives in rnext/planner.py and is a CONSTANT
   of the process, so it cannot differ between two scorings).
   count[c] is how many times candidate c has been scored. *)

CONSTANT Candidates, Scores

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

Next == ScoreOne

Spec == Init /\ [][Next]_<<pending, ranked, count>>

TypeOK ==
    /\ pending \subseteq Candidates
    /\ count \in [Candidates -> 0..2]
    /\ ranked \in [Candidates -> 0..100000]

(* каждый кандидат оценивается не более одного раза за прогон *)
NoDoubleScoring == \A c \in Candidates: count[c] <= 1

(* оценка кандидата берётся только из единственной таблицы Scores *)
ScoredSingleTable == \A c \in Candidates: count[c] = 1 => ranked[c] = Scores[c]

====
