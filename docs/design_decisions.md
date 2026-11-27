# Architecture and Design Decisions

## DD001 — Configuration Management

**Boundary:** Configuration ↔ Application

**Contract:** Structured YAML configuration accessed through ConfigManager.

**What can vary:**
- Environment parameters
- Collection frequency
- Task parameters
- Storage locations
- Training and evaluation parameters

**Why:** Run different experiments without modifying application code and preserve experiment reproducibility.

**V0:** Local YAML files with validation and configuration cloning.

## DD002 — Environment and Action

**Boundary:** Action producer ↔ Environment

**Contract:** Action → EnvironmentState.

**What can vary:**
- Human or policy-based action generation
- Environment dynamics
- Action representation
- Physical or simulated embodiment

**Why:** Environment state transitions should be independent of the source of actions.

**V0:** Deterministic cursor movement, boundary clamping and mouse button state.

## DD003 — Environment and Task

**Boundary:** Environment ↔ Task

**Contract:** Previous state + Action + Current state → TaskResult.

**What can vary:**
- Task objectives
- Success criteria
- Task complexity
- Evaluation metrics

**Why:** Multiple tasks can operate within the same environment without modifying its state transition logic.

**V0:** Stateless stationary-target click evaluation.

## DD004 — Episode Orchestration

**Boundary:** Episode ↔ Environment and Task

**Contract:** Initialize, start, step, check timeout, abort and retrieve metadata.

**What can vary:**
- Environment implementation
- Task implementation
- Action producer
- Episode duration
- Clock implementation

**Why:** Execution lifecycle should be independent of rendering, data acquisition and persistence.

**V0:** Synchronous execution, externally supplied actions and an injectable monotonic clock.

## DD005 — Data Acquisition

**Boundary:** Data producer ↔ Recorder

**Contract:** Named streams containing timestamped records conforming to predefined schemas.

**What can vary:**
- Data producer
- Number and type of streams
- Sampling frequency
- Data modality
- Stream schema

**Why:** Recording infrastructure should be reusable across human demonstrations, policy execution, simulation and physical robots.

**V0:** Three streams: RGB observations, asynchronous actions and environment state.

## DD006 — Data Persistence

**Boundary:** Recorder ↔ Storage

**Contract:** Validated records passed to stream-specific serialization and storage handlers.

**What can vary:**
- File formats
- Compression
- Storage backend
- Buffering strategy
- Synchronous or asynchronous writes

**Why:** Storage optimizations should not require changes to data producers or episode execution.

**V0:** Local filesystem, PNG images, CSV records and JSON metadata. Synchronous writes initially.

**Limitation:** Impact of image encoding and disk writes on acquisition timing remains unmeasured.

## DD007 — Recording Lifecycle

**Boundary:** Episode ↔ Recorder

**Contract:** Start and stop recording with episode identity and metadata.

**What can vary:**
- Recording enabled or disabled
- Recorder implementation
- Registered streams
- Persistence strategy

**Why:** Episode execution should control the recording lifecycle without knowing what data is being recorded or how it is stored.

**V0:** Optional recorder dependency with synchronous lifecycle control.

**Required guarantee:** Initial observation, terminal action and terminal observation must be preserved.

**Unresolved:** Ordering mechanism for terminal data delivery before recorder shutdown.

## DD009 — Input Abstraction

**Boundary:** Physical input source ↔ Application action space.

**Contract:** Pygame Event → Action or None.

**What can vary:**
- Input hardware
- Input library
- Control mapping
- Human or autonomous action generation

**Why:** Environment execution should be independent of the input hardware and event-generation mechanism.

**V0:** Stateless translation of relative mouse movements and left-button events.

**Limitations:** No input smoothing, sensitivity adjustment or additional button mappings.