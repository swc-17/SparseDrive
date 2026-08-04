"""NAVSIM stage 2 vocab model, extended schedule: 10 epochs total.

Continuation of sparsedrive_navsim_stage2_vocab_full.py (which trained 3
epochs to iter_8703 with losses still descending). Submitted with the same
run_name so the Lilypad entrypoint auto-resumes from the epoch-3 checkpoint;
cosine LR then anneals over the new 10-epoch horizon.
"""

_base_ = ["./sparsedrive_navsim_stage2_vocab_full.py"]

# 92,853 navtrain-train samples / total batch 32 = 2,901 iters per epoch,
# identical to the base schedule; only the epoch count changes.
num_iters_per_epoch = 2901
num_epochs = 10

runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)
checkpoint_config = dict(interval=num_iters_per_epoch)
