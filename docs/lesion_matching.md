# Lesion Matching with NetworkX Min-Cost Flow

## Run

### 1. Generate lesion pair costs and cost matrix

```powershell
python tools/generate_lesion_pair_costs.py `
  --features "outputs\aligned_lesion_features.csv" `
  --pet-feature none `
  --pet-weight 0 `
  --out-pairs "outputs\lesion_pair_costs.csv" `
  --matrix-dir "outputs\lesion_pair_cost_matrices"
```

Example matrix output:

```text
outputs/
└── lesion_pair_cost_matrices/
    └── 0a09c8844b_cost_matrix.csv
```

### 2. Run the min-cost-flow matcher directly from the cost matrix

```powershell
python tools/run_lesion_min_cost_flow.py `
  --cost-matrix "outputs\lesion_pair_cost_matrices\0a09c8844b_cost_matrix.csv" `
  --disappearing-penalty 1.0 `
  --new-lesion-penalty 1.0 `
  --merge-penalty 0.2 `
  --max-bl-per-fu 3 `
  --cost-scale 1000 `
  --out "outputs\lesion_matches.csv" `
  --graph-dir "outputs\lesion_flow_graphs"
```

### 3. Run matcher tests

```powershell
python -m pytest -q tests/test_lesion_min_cost_flow.py
```

Current matcher tests:

```text
13 passed
```

---

# Scope

This document only describes the work from the completed **BL × FU lesion cost matrix** to the completed **automatic lesion correspondence matcher**.

The earlier stages such as CT loading, rigid registration, mask transformation, and aligned-lesion feature extraction are not covered here.

The implemented section of the pipeline is:

```text
Aligned lesion features
        ↓
Generate BL × FU pair costs
        ↓
Export labelled cost matrix
        ↓
Load cost matrix
        ↓
Build min-cost-flow network
        ↓
NetworkX min_cost_flow
        ↓
Decode selected flow
        ↓
MATCHED / DISAPPEARING / NEW / MERGING
        ↓
lesion_matches.csv
```

---

# 1. Cost Matrix

The pair-cost stage produces one labelled BL × FU matrix for each patient.

Rows represent Baseline lesions and columns represent Follow-up lesions.

Example:

```text
                         FU_L001    FU_L002
BL_L001                   4.107       0.819
BL_L002                   4.203       0.767
BL_L003                   0.127       4.527
BL_L004                   4.685       0.492
```

A lower value means the BL/FU pair is considered more similar under the current pair-cost model.

The matrix is produced by:

```text
tools/generate_lesion_pair_costs.py
```

and exported to:

```text
outputs/lesion_pair_cost_matrices/
```

For example:

```text
outputs/lesion_pair_cost_matrices/0a09c8844b_cost_matrix.csv
```

The matcher can now read this matrix directly.

---

# 2. Matrix Loader

The matcher includes a matrix loader that converts the labelled matrix into candidate graph edges.

For each finite matrix cell:

```text
BL lesion → FU lesion
```

a candidate match is created with the matrix value as its pair cost.

For example:

```text
BL_L003 → FU_L001
cost = 0.1268
```

becomes a candidate edge in the matching graph.

If a matrix cell contains:

```text
inf
```

the pair is treated as unavailable and no BL → FU candidate edge is created.

However, the BL and FU lesion IDs are still preserved. This is important because a lesion with no remaining candidate can still become:

```text
DISAPPEARING
```

or:

```text
NEW
```

during matching.

---

# 3. Why Minimum-Cost Flow Is Used

Selecting the lowest-cost FU candidate independently for every BL lesion is not sufficient because multiple BL lesions may compete for the same FU lesion.

For example:

```text
BL_L001 → FU_L002    0.819
BL_L002 → FU_L002    0.767
BL_L004 → FU_L002    0.492
```

A local ranking cannot decide whether:

```text
only one BL should match FU_L002
```

or:

```text
multiple BL lesions should merge into FU_L002
```

The matcher therefore constructs one global optimisation problem and solves it using:

```python
networkx.min_cost_flow(...)
```

NetworkX chooses the feasible set of assignments with the lowest total cost under the graph constraints.

---

# 4. Matcher Structure

The matcher is divided into three main parts.

```text
src/lesion_min_cost_flow.py

PART 1
Build matching network
        ↓
PART 2
Run NetworkX min_cost_flow
        ↓
PART 3
Decode selected flow
```

This separation is intentional.

The lesion-specific graph design is kept separate from the generic NetworkX solver, and the raw flow result is then converted back into lesion-level clinical events.

---

# 5. Part 1 — Build the Matching Network

## 5.1 BL lesions

Each BL lesion supplies one unit of flow.

Conceptually:

```text
BL_L001  supply 1
BL_L002  supply 1
BL_L003  supply 1
...
```

This ensures one BL lesion can only receive one final outcome.

Its possible outcomes are:

```text
BL → FU primary match
BL → merge event
BL → DISAPPEAR
```

---

## 5.2 Normal BL → FU match

A normal match is represented by a primary FU slot:

```text
BL → FU_PRIMARY
```

The edge cost is the original pair cost from the matrix.

Example:

```text
BL_L003 → FU_L001
pair cost = 0.1268
```

If selected, this is decoded as:

```text
MATCHED
```

provided that the FU is not part of a multi-BL merge event.

---

# 6. Unmatched Events

The matcher explicitly represents unmatched lesions.

## 6.1 DISAPPEARING

Every BL lesion has the option:

```text
BL → DISAPPEAR
```

with:

```text
disappearing_penalty
```

For example:

```text
disappearing_penalty = 1.0
```

If matching a BL lesion to an FU lesion costs more than disappearing, the solver may choose:

```text
DISAPPEARING
```

instead of forcing an unrealistic match.

---

## 6.2 NEW

Every FU lesion can also be explained by:

```text
NEW → FU_PRIMARY
```

with:

```text
new_lesion_penalty
```

For example:

```text
new_lesion_penalty = 1.0
```

If no suitable BL lesion is selected for that FU lesion, the result can be decoded as:

```text
NEW
```

---

# 7. MERGING as a Separate Event

MERGING is deliberately represented as a separate graph event instead of simply increasing FU capacity.

This makes the merge logic easier to inspect and easier to improve later.

The graph structure is conceptually:

```text
BL
 │
 │ pair cost
 ▼
MERGE_EVENT for FU
 │
 │ merge event penalty
 ▼
FU_MERGE_SLOT
```

For example:

```text
BL_L002
   │
   │ pair cost = 0.7673
   ▼
MERGE_EVENT::FU_L002
   │
   │ merge penalty = 0.2
   ▼
FU_L002 merge slot
```

The total cost for this merge assignment is:

```text
0.7673 + 0.2 = 0.9673
```

This is kept separate from a normal primary assignment.

---

# 8. Merge Slots

The parameter:

```text
max_bl_per_fu
```

controls how many BL lesions may finally correspond to one FU lesion.

Examples:

```text
max_bl_per_fu = 1
```

means:

```text
one-to-one matching only
```

No merge is possible.

---

```text
max_bl_per_fu = 2
```

means:

```text
1 PRIMARY slot
+
1 MERGE slot
```

so at most two BL lesions can correspond to one FU lesion.

---

```text
max_bl_per_fu = 3
```

means:

```text
1 PRIMARY slot
+
2 MERGE slots
```

so at most three BL lesions can correspond to one FU lesion.

Unused merge slots are filled by zero-cost balancing flow. They do not represent real NEW lesions and have no biological meaning.

---

# 9. Merge Penalty

The parameter:

```text
merge_penalty
```

controls how strongly the model discourages additional BL lesions from merging into the same FU lesion.

For a merge assignment:

```text
merge match cost
=
pair cost
+
merge penalty
```

For the current patient:

```text
BL_L002 → FU_L002
pair cost = 0.7673
```

With:

```text
merge_penalty = 0.2
```

the merge cost becomes:

```text
0.7673 + 0.2
= 0.9673
```

Since:

```text
0.9673 < disappearing penalty 1.0
```

the solver prefers the merge.

With:

```text
merge_penalty = 0.3
```

the cost becomes:

```text
0.7673 + 0.3
= 1.0673
```

Now:

```text
1.0673 > disappearing penalty 1.0
```

so the solver prefers:

```text
DISAPPEARING
```

instead.

This confirms that the merge event participates directly in the global minimum-cost decision.

---

# 10. Current Patient Example

For patient:

```text
0a09c8844b
```

the important candidate costs are:

```text
BL_L003 → FU_L001 = 0.1268

BL_L004 → FU_L002 = 0.4917
BL_L002 → FU_L002 = 0.7673
BL_L001 → FU_L002 = 0.8186
```

Using:

```text
disappearing_penalty = 1.0
new_lesion_penalty   = 1.0
merge_penalty        = 0.2
max_bl_per_fu        = 3
```

the matcher selects:

```text
BL_L001 → DISAPPEARING

BL_L003 → FU_L001
           MATCHED

BL_L002 ─┐
         ├── FU_L002
BL_L004 ─┘
           MERGING
```

The relevant costs are:

```text
BL_L001 → DISAPPEARING
= 1.0

BL_L003 → FU_L001
= 0.1268

BL_L004 → FU_L002 PRIMARY
= 0.4917

BL_L002 → FU_L002 MERGE
= 0.7673 + 0.2
= 0.9673
```

The total objective is approximately:

```text
1.0
+ 0.1268
+ 0.4917
+ 0.9673
=
2.5858
```

---

# 11. Part 2 — NetworkX Solver

After the graph is built, the optimisation itself is delegated to NetworkX.

Conceptually:

```python
flow = nx.min_cost_flow(
    graph,
    demand="demand",
    capacity="capacity",
    weight="weight",
)
```

NetworkX does not understand lesion tracking directly.

It only solves the graph defined by the project code.

The project is responsible for defining:

```text
lesion nodes
candidate edges
NEW event
DISAPPEARING event
MERGE event
slot limits
edge capacities
edge weights
node demands
```

NetworkX is responsible for finding the feasible flow with the minimum total cost.

---

# 12. Integer Cost Scaling

Pair costs are floating-point values such as:

```text
0.1267919876
```

Before being passed to NetworkX they are scaled into integer edge weights.

For example:

```text
cost_scale = 1000
```

gives:

```text
0.126792 → 127
0.491721 → 492
1.000000 → 1000
```

The original floating-point costs are preserved separately for final reporting.

---

# 13. Part 3 — Decode the Flow

NetworkX returns selected graph flows rather than lesion-level labels.

The decoder converts these selected edges into:

```text
MATCHED
DISAPPEARING
NEW
MERGING
```

For normal one-to-one flow:

```text
BL → FU_PRIMARY
```

the result is:

```text
MATCHED
```

For:

```text
BL → DISAPPEAR
```

the result is:

```text
DISAPPEARING
```

For:

```text
NEW → FU_PRIMARY
```

the result is:

```text
NEW
```

For multiple BL lesions assigned to the same FU through a primary slot and merge slots:

```text
BL_A ─┐
      ├── FU
BL_B ─┘
```

all lesion rows involved are labelled:

```text
MERGING
```

The internal PRIMARY role is not interpreted as a separate biological event.

---

# 14. Merge Event ID

All BL lesions that participate in the same merge share a common event ID.

Example:

```text
MERGE::0a09c8844b::0a09c8844b_FU_L002
```

This allows the final output to represent:

```text
BL_L002 ─┐
         ├── FU_L002
BL_L004 ─┘
```

as one explicit merge event.

Keeping this event ID is useful for later improvements such as:

```text
combined BL volume
merge-specific geometry
PET consistency
merge confidence
event-level evaluation
```

---

# 15. Final Output

The final structured result is written to:

```text
outputs/lesion_matches.csv
```

Important columns include:

| Column | Meaning |
|---|---|
| `patient_id` | Patient identifier |
| `bl_lesion_id` | Baseline lesion ID |
| `fu_lesion_id` | Follow-up lesion ID |
| `match_type` | `MATCHED`, `DISAPPEARING`, `NEW`, or `MERGING` |
| `assignment_role` | Internal assignment role such as `PRIMARY`, `MERGE`, or `DISAPPEAR` |
| `event_id` | Shared MERGING event identifier when relevant |
| `pair_cost` | Original BL/FU candidate cost |
| `event_cost` | Additional event cost such as merge penalty |
| `match_cost` | Total cost associated with the decoded assignment |

Example:

```text
BL_L001 → DISAPPEARING
match_cost = 1.0

BL_L003 → FU_L001
MATCHED
match_cost = 0.1268

BL_L002 → FU_L002
MERGING
pair_cost = 0.7673
event_cost = 0.2
match_cost = 0.9673

BL_L004 → FU_L002
MERGING
pair_cost = 0.4917
event_cost = 0
match_cost = 0.4917
```

---

# 16. Flow Graph Debug Output

The CLI can also export the graph edges to:

```text
outputs/lesion_flow_graphs/
```

For example:

```text
outputs/lesion_flow_graphs/0a09c8844b_flow_graph_edges.csv
```

This file contains fields such as:

```text
source
target
edge_type
capacity
weight
original_cost
base_pair_cost
event_cost
selected_flow
```

A row with:

```text
selected_flow = 1
```

means that NetworkX selected that graph edge.

This provides a direct way to inspect whether a result came from:

```text
normal MATCH
DISAPPEARING
NEW
MERGE_CANDIDATE
MERGE_EVENT
unused balancing slot
```

---

# 17. Parameter Behaviour Checks

The current implementation was checked with several configurations.

## Merge disabled

```text
max_bl_per_fu = 1
```

Result:

```text
BL_L001 → DISAPPEARING
BL_L002 → DISAPPEARING
BL_L003 → FU_L001
BL_L004 → FU_L002
```

---

## Merge allowed but expensive

```text
max_bl_per_fu = 3
merge_penalty = 0.3
```

Result remains:

```text
BL_L001 → DISAPPEARING
BL_L002 → DISAPPEARING
BL_L003 → FU_L001
BL_L004 → FU_L002
```

The graph allows merging, but the solver decides it is too expensive.

---

## Two-to-one merge

```text
max_bl_per_fu = 3
merge_penalty = 0.2
```

Result:

```text
BL_L001 → DISAPPEARING

BL_L002 ─┐
         ├── FU_L002
BL_L004 ─┘

BL_L003 → FU_L001
```

---

## Three-to-one merge

```text
max_bl_per_fu = 3
merge_penalty = 0.1
```

Result:

```text
BL_L001 ─┐
BL_L002 ─┼── FU_L002
BL_L004 ─┘

BL_L003 → FU_L001
```

---

## Slot limit

With:

```text
max_bl_per_fu = 2
merge_penalty = 0.1
```

FU_L002 can accept at most:

```text
1 PRIMARY
+
1 MERGE
```

so only two BL lesions are allowed to map to FU_L002.

This confirms that `max_bl_per_fu` is a real optimisation constraint, not only an output setting.

---

# 18. Current Status

The completed matching section now supports:

```text
✓ labelled BL × FU cost matrix export
✓ direct cost-matrix loading
✓ candidate edge construction
✓ minimum-cost-flow graph construction
✓ NetworkX min_cost_flow solving
✓ one-to-one matching
✓ DISAPPEARING lesions
✓ NEW lesions
✓ explicit MERGE events
✓ configurable merge penalty
✓ configurable merge slot limit
✓ structured lesion_matches.csv output
✓ flow graph debug export
✓ automated matcher tests
```

The current MERGING implementation is intentionally a baseline.

The merge event is already separated from the normal pair cost so that the fixed merge penalty can later be replaced or extended by a more specific event model without redesigning the entire matcher.
