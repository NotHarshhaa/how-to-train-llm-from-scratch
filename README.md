# Train LLM From Scratch — Interactive Companion

An interactive, zero-dependency educational website for [NotHarshhaa/how-to-train-llm-from-scratch](https://github.com/NotHarshhaa/how-to-train-llm-from-scratch) explaining the from-scratch journey from raw text, tokenization, architecture, and next-token cross-entropy loss to SFT, RM, DPO, PPO, and GRPO.

## Credits & Acknowledgements
This site is completely based on the incredible repository and work by **Fareed Khan**:
👉 **[https://github.com/FareedKhan-dev/train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch)**

All core architectural breakdowns, mathematical equations, loss data endpoints, and training methodologies are derived from Fareed Khan's original project.

## Features
- **The Source-Line**: Step-through interactive pipeline stepper walking through the 11 stages from raw text to chat.
- **Live Tokenizer**: Split text with the GPT-2 BPE regex and inspect stand-in token IDs.
- **Model Parameter Calculator**: Exact parameter formula $\tau(d, N, T, V)$ with 13M, 77M, and 406M presets plus custom dimension validation.
- **The Five Seats**: Code and mathematical formulation of MLP, Head, Multi-Head, Block, and Transformer.
- **Next-Token Loss & Generation**: Interactive training step scrubber (0–2000 steps) with dynamic loss curves and Markov text generation.
- **Post-Training Alignment**: SFT, RM, DPO, PPO, and GRPO formulation breakdown with theorems T1–T9.
- **GSM8K Evaluator**: Greedy answer parser with `<answer>` tag detection and gold validation.
- **Fast & Zero-Dependency**: Written entirely in pure HTML5, CSS3, and modern Vanilla JavaScript. Zero build step, zero Node.js runtime required.

## Live Site
👉 **[https://notharshhaa.github.io/how-to-train-llm-from-scratch/](https://notharshhaa.github.io/how-to-train-llm-from-scratch/)**
