# Target-Adaptive Selection Study

This supporting study evaluates configuration choices made using pseudo-missing validation tasks on support maps, then measures their performance on the known Acp_4001 masked regions. It is included because the paper discusses automated selection and its limitations. It is intentionally separate from the primary seven-case reproduction scripts.

## Run

Install the handover's pinned requirements, then run:

```powershell
cd standalone\ISABE2026\methodology_experiments
python isabe2026_target_adaptive_configuration_study.py
```

This directory mirrors the relative layout expected by the study scripts. The run writes its outputs under `standalone/ISABE2026/methodology_experiments/target_adaptive_selection_study_outputs/`. A reference snapshot is preserved in `saved_results/`.

The archived comparison outputs show close agreement for lower exterior extrapolation, but larger errors for some cases, particularly upper exterior extrapolation. This supports describing automated selection as a useful recommendation rather than a guarantee of the target-case optimum. See the paper and `../../HANDOVER_FOR_AI.md` for context and interpretation.
