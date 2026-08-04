"""NAVSIM evaluation stack for original SparseDrive (Phase 4a/4b).

Layout (two-environment design; no single local env has both the mm-stack
and the navsim/nuplan stack):

- ``coord.py``               pure-numpy SD<->NAVSIM coordinate contract,
                             heading derivation, stationary fallback;
- ``agent_input_adapter.py`` NAVSIM ``AgentInput`` -> SparseDrive pipeline
                             input dicts (duck-typed; no navsim import);
- ``runner.py``              SparseDrive checkpoint loader + per-scenario
                             temporal-bank reset + 4-frame history replay
                             (needs the SparseDrive env, e.g. sparsedrive310);
- ``agent.py``               the NAVSIM ``AbstractAgent`` wrapper proper
                             (needs an env with BOTH stacks; see runbooks);
- ``run_inference_infos.py`` offline inference over converted navmini infos
                             -> {token: (8,3) NAVSIM poses} pickle
                             (sparsedrive310 env);
- ``score_pdm_v1.py``        NAVSIM v1 PDMS scoring inside SparseDriveV2's
                             env (lilypad) with its pinned metric caches;
- ``score_epdms_two_stage.py`` NAVSIM v2 two-stage EPDMS scoring inside
                             SparseDriveV2's env (lilypad).

Runbooks: docs/navsim_eval_openloop.md, docs/navsim_eval_closedloop.md.
"""
