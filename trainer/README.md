# Training tools

Documentation index only. Working implementations retain their existing imports.

| Task | Implementation / instructions |
| --- | --- |
| Synthetic invoice annotation and SFT | [src/trainer/sft](../src/trainer/sft/) |
| Real CORD preparation (161/20/19) | [CORD preparation](../src/trainer/cord/README.md) |
| CORD multimodal QLoRA | [CORD training](../src/trainer/cord/TRAINING.md) |
| Reasoning preferences (60/10/10) | [Preference annotation](../src/trainer/preference_optimization/README.md) |
| Text-model DPO + QLoRA | [DPO training](../src/trainer/preference_optimization/train/README.md) |

Run modules from the repository root in `../.venv`. Stop both GPU servers before
any training. Extraction and reasoning adapters remain separate. No PPO, reward
model or automatic retraining. See [model artifacts](../models/README.md).
