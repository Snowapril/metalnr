# AGENTS.md — Metal Cloud Renderer

## Project Goal

Build a research-oriented volumetric cloud renderer for macOS using Metal 4.

Implement and compare three rendering techniques:

1. Path-traced reference rendering
2. Analytic volumetric rendering
3. Neural volumetric rendering

The primary goals are correctness, reproducibility, visual quality, GPU performance, and maintainability.

This is an experimental renderer, not a general-purpose game engine.

## Engineering Principles

- Prefer simple, explicit, maintainable solutions.
- Apply the installed Ponytail skill throughout development.
- Understand existing code before changing it.
- Reuse existing code, standard libraries, and native platform APIs.
- Avoid speculative abstractions, unnecessary dependencies, and premature optimization.
- Do not introduce a general-purpose rendering framework.
- Optimize only when correctness is established and measurements justify the change.
- Never sacrifice rendering correctness or benchmark integrity to reduce code size.

## Development Workflow

For each nontrivial task:

1. Inspect the relevant code and documentation.
2. Identify existing functionality that can be reused.
3. State the intended change and acceptance criteria.
4. Implement the smallest complete solution.
5. Build and run relevant tests.
6. Inspect warnings and Metal validation errors.
7. Summarize changes, verification results, and remaining limitations.

For large tasks, first propose a short implementation plan.

Do not claim successful compilation, GPU execution, or validation without actually performing it.

Do not silently substitute placeholder rendering results for real implementations.

## Technology

- Target: macOS 26+ on Apple Silicon.
- Graphics API: Metal 4.
- Host code: C++20 and Objective-C++.
- GPU shaders: Metal Shading Language.
- Neural training: Python/PyTorch when needed.
- Neural inference: Metal GPU execution.
- Prefer native Apple APIs over additional dependencies.

Verify API availability against the installed SDK before implementation.

Do not invent Metal API names, capabilities, or performance guarantees.

## Architecture

Maintain a clear separation between:

- Application and window management
- Metal device and resource management
- Scene and cloud representation
- Rendering algorithms
- Neural inference and model assets
- Benchmarking and validation

The three rendering methods must share the same scene definition.

Avoid duplicating scene, camera, lighting, and density logic.

Do not force algorithm-specific data into an unnecessarily generic renderer interface.

## Shared Scene Contract

All rendering methods must use compatible definitions of:

- Camera position, orientation, projection, and exposure
- World coordinate system and units
- Cloud density field
- Cloud volume boundaries
- Extinction and scattering coefficients
- Single-scattering albedo
- Phase function and anisotropy
- Lighting and environment
- Output resolution and color space

Document any approximation or unsupported feature.

Changes to shared physical definitions must be reflected in tests and documentation.

## Rendering Correctness

The path tracer serves as the reference implementation.

- Use documented radiative transfer assumptions.
- Track stochastic seeds and sample counts.
- Make random sequences reproducible.
- Support progressive accumulation.
- Do not compare unconverged reference images as ground truth.
- Document known bias and variance.

The analytic renderer must document its approximations.

The neural renderer must document its training targets, inputs, outputs, and inference precision.

## Neural Rendering

Keep training and inference responsibilities separate.

- Do not require Python for normal renderer execution.
- Use a versioned model artifact.
- Record model architecture and training configuration.
- Record training dataset and reference renderer versions.
- Validate model input/output conventions.
- Never use evaluation scenes as training scenes unless explicitly permitted by the experiment protocol.
- Report model loading separately from steady-state inference.

Do not add an ML framework to the runtime without demonstrating a concrete need.

## GPU Engineering

- Use Metal resource and synchronization APIs correctly.
- Respect buffer alignment and resource lifetime requirements.
- Avoid unnecessary CPU-GPU synchronization.
- Avoid blocking the CPU for GPU results in the render loop.
- Make resource ownership explicit.
- Check Metal validation errors.
- Distinguish CPU encoding time from GPU execution time.
- Measure before introducing asynchronous compute or advanced scheduling.

Prefer clear GPU algorithms over speculative low-level optimization.

## Benchmark Protocol

All methods must support identical benchmark scenes.

Record:

- Renderer and configuration
- Scene and camera identifiers
- Resolution
- GPU and OS version
- Sample count or quality settings
- GPU rendering time
- CPU frame preparation time
- Memory usage where measurable
- Image quality metrics
- Model version for neural rendering

Use GPU execution timing for GPU comparisons.

Exclude shader compilation, resource initialization, model loading, and reference accumulation from steady-state timing unless explicitly measuring those costs.

Report these excluded costs separately when relevant.

Do not compare rendering methods at different quality settings without clearly labeling the comparison.

## Testing

Maintain tests for:

- Camera and coordinate transformations
- Density field sampling
- Phase function behavior
- Transmittance and extinction
- Renderer configuration serialization
- Deterministic benchmark configurations
- Image comparison and metric calculation

Use GPU integration tests for shader behavior when possible.

For performance-sensitive changes, compare before and after measurements.

Never replace a failing test with a weaker test merely to make it pass.

## Documentation

Keep documentation concise and current.

Document:

- Architectural decisions
- Rendering equations and approximations
- Build and execution instructions
- Benchmark methodology
- Known limitations

Update documentation when public interfaces or rendering assumptions change.

## Code Style

- Prefer RAII and explicit ownership.
- Prefer value semantics where practical.
- Avoid global mutable state.
- Avoid unnecessary inheritance and template complexity.
- Keep shader and host data layouts synchronized.
- Use descriptive names and consistent units.
- Comments should explain non-obvious reasoning, not restate code.

## Scope Control

Do not implement features outside the current milestone.

In particular, avoid adding the following without an explicit requirement:

- General-purpose ECS
- Plugin architecture
- Custom scripting language
- Custom memory allocator
- General-purpose render graph
- Complex editor UI
- Unnecessary third-party libraries

## Completion Report

At the end of each task, report:

1. What changed
2. What was verified
3. What could not be verified
4. Remaining risks or limitations
5. Suggested next step

Keep the report concise and factual.