R01 — Inter-operator variability
- Compare demonstrations from different human operators.
- Study differences in movement speed, precision, trajectory efficiency and corrective behavior.
- Investigate whether learned policies inherit their demonstrators' behavioral characteristics.

R02 — Intentional demonstration diversity
- Natural versus deliberately introduced behavioral diversity.
- Operator instructions as a specification of desirable behaviors.
- Useful diversity versus arbitrary or undesirable movements.
- Impact of demonstration diversity on policy generalization.

R03 — Controlled scenario sampling
- Generate predefined initial cursor positions, target positions and target sizes.
- Reuse identical scenarios across operators and sessions.
- Stratified sampling to ensure systematic coverage.
- Compare different strategies under identical initial conditions.
- Introduce scenario_id for traceability.

R04 — Partial observability
- Restrict the operator's field of view.
- Introduce exploration, discovery, navigation and execution phases.
- Study systematic versus random exploration strategies.
- Separate ground-truth state from observable information.

R05 — Stochastic motion
- Introduce Gaussian noise during movement.
- Study corrective and adaptive behaviors.
- Investigate policy robustness under unfamiliar disturbance levels.
- Distinguish commanded actions from actual applied actions.

R06 — Variable latency
- Introduce random action latency.
- Introduce random observation latency.
- Study how operators adapt to delayed feedback.
- Investigate the importance of temporal history in policy learning.

R07 — Combined uncertainty
- Combine partial observability, motion noise and latency.
- Study generalization and robustness under increasingly realistic operating conditions.