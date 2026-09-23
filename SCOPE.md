# Reproducibility scope

This anonymous package is intentionally limited to experiments and analyses reported in the FormalCRRC manuscript. 

## Included manuscript evidence

| Manuscript component | Reproducibility paths |
|---|---|
| Synthetic S1-S3 threshold studies | `studies/legacy`, `code/src/formalcrrc`, selected `code/scripts` |
| Artifact-blind baselines and reachability theory | `code/src/formalcrrc/baselines.py`, `step_proposition.py`, analysis scripts |
| Label-swap study | `studies/legacy`, label-swap modules/scripts |
| Qwen3 synthetic reasoning anchor | `studies/legacy`, `protocols/qwen3_synthetic`, reasoning-anchor modules/scripts |
| Qwen2.5 synthetic reason-then-score ablation | `studies/synthetic_rts`, `protocols/synthetic_rts`, `rts_ablation*` source/tests |
| Historical HumanEval+/code panel and Qwen3 continuation | `studies/legacy`, `protocols/code_panel`, code-panel modules/scripts |
| Corrected same-engine code protocol and structural controls | `studies/code_controls`, `protocols/code_v2`, `inference/code_protocol_followup_v2` |
| Earlier 14-task extension | `studies/replication14`, `inference/table8_independent_replication_v1` |
| Post-hoc held-out calibration | `studies/calibration` |
| Independent 276-task MBPP study, count control, interactions, cost/behavior summaries | `studies/mbpp`, `inference/mbpp`, `protocols/templates.json` |

## Deliberately excluded

- Author names, personal/workstation paths, SSH configuration, institution-specific cluster connection instructions, and local account names.
- API keys, tokens, private keys, `.env` files, credential files, caches, virtual environments, and generated logs.
- Site-specific cluster submission/probe/sync helpers that are not required to reconstruct manuscript results.
- Model weights and third-party benchmark bodies/reference solutions/enhanced test bodies that may be redistribution-restricted.

Saved numerical evidence is retained where it is sufficient to reproduce the reported analyses without rerunning GPU inference. Fresh end-to-end inference requires upstream datasets/models and adaptation to the user's own compute environment.
