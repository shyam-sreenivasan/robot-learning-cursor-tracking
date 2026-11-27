## Data Validation — Future Improvements

- Negative unit tests for individual validation checks.
- Execution replay validation using temporary episodes.
- Systematic acquisition fault injection study.
- Measure fault detection rates and false positives.
- Image-to-state consistency validation.
- Optimize validation performance and redundant disk reads.
- Establish acceptable acquisition quality thresholds based on empirical observations.



## Terminology and Namespace Refactoring

**Objective:** Separate data collection terminology from learning experimentation terminology.

### Data Acquisition Hierarchy
Collection Protocol → Session → Episode

### Policy Learning Hierarchy
Dataset → Experiment → Run → Checkpoint → Evaluation

### Pending Changes
- Rename collection experiment to collection protocol.
- Rename configuration files and associated identifiers.
- Update DataCatalog interfaces and metadata schemas.
- Update Flask API endpoints and dashboard terminology.
- Plan backward-compatible migration of existing recordings.
- Preserve configuration hashes, provenance and historical metadata.

### Design Constraints
- A dataset can contain episodes from multiple collection protocols.
- A dataset can support multiple learning experiments.
- An experiment can have multiple independent runs.
- Each run preserves its configuration snapshot, checkpoints and results.

**Priority:** Low.

**Implementation:** Deferred until after establishing the first end-to-end learning baseline.