import { i as __toESM } from "../_runtime.mjs";
import { L as require_react, v as require_jsx_runtime } from "../_libs/@tanstack/react-router+[...].mjs";
import { n as Copy, r as Check } from "../_libs/lucide-react.mjs";
import { t as clsx } from "../_libs/clsx.mjs";
import { t as twMerge } from "../_libs/tailwind-merge.mjs";
//#region node_modules/.nitro/vite/services/ssr/assets/routes--_HTFy4d.js
var import_react = /* @__PURE__ */ __toESM(require_react());
var import_jsx_runtime = require_jsx_runtime();
var REPO = "https://github.com/FareedKhan-dev/train-llm-from-scratch";
var DOCS = "https://fareedkhan-dev.github.io/train-llm-from-scratch/";
var PAPER = "https://arxiv.org/abs/1706.03762";
var AUTHOR = "https://github.com/FareedKhan-dev";
var PIPELINE = [
	{
		id: "text",
		label: "raw text",
		hint: "The Pile, instructions, preferences",
		sink: false
	},
	{
		id: "tokens",
		label: "tokens",
		hint: "tiktoken r50k_base → HDF5",
		sink: false
	},
	{
		id: "tau",
		label: "Transformer",
		hint: "MLP · Head · Block",
		sink: false
	},
	{
		id: "ce",
		label: "CE loss",
		hint: "next-token cross-entropy",
		sink: false
	},
	{
		id: "base",
		label: "base",
		hint: "a language model",
		sink: false
	},
	{
		id: "sft",
		label: "SFT",
		hint: "assistant tokens only",
		sink: false
	},
	{
		id: "rm",
		label: "RM",
		hint: "Bradley-Terry",
		sink: false
	},
	{
		id: "dpo",
		label: "DPO",
		hint: "no reward model",
		sink: false
	},
	{
		id: "ppo",
		label: "PPO",
		hint: "clipped surrogate",
		sink: false
	},
	{
		id: "grpo",
		label: "GRPO",
		hint: "group-relative",
		sink: false
	},
	{
		id: "chat",
		label: "chat",
		hint: "the sink",
		sink: true
	}
];
var THEOREMS = [
	{
		id: "T1",
		title: "τ is assembled from small pieces",
		body: "MLP, a causal Head, MultiHeadAttention, a pre-norm Block, then Embedding + N blocks + LayerNorm + lm_head. Each file is a few dozen lines of plain PyTorch. There is no transformers.Transformer."
	},
	{
		id: "T2",
		title: "Pretraining is next-token CE",
		body: "Write logits over the vocabulary at every position, reshape, and call F.cross_entropy against the shifted tokens. The 77M run in the README falls from 11.14 train loss to 3.73 / 3.76 dev in 2 000 steps."
	},
	{
		id: "T3",
		title: "SFT masks everything except the assistant",
		body: "Chat turns are packed with <|user|>, <|assistant|>, <|endoftext|>. A loss_mask zeros the user tokens so the model is graded only on the reply: ce * mask; return ce.sum() / mask.sum()."
	},
	{
		id: "T4",
		title: "The reward model is Bradley-Terry",
		body: "A scalar head on the same backbone scores (chosen, rejected) pairs. Loss is −log σ(chosen − rejected). The README reports 0.574 accuracy on 7 974 held-out pairs — a weak but real preference signal."
	},
	{
		id: "T5",
		title: "DPO aligns without a reward model",
		body: "DPO: −log σ(β (π log-ratio − π_ref log-ratio)). ORPO is reference-free and folds SFT into the same step. KTO takes unpaired signal. Same small Transformer, real preference data (HH-RLHF / UltraFeedback)."
	},
	{
		id: "T6",
		title: "PPO is the clipped surrogate plus GAE",
		body: "ratio = exp(new_logp − old_logp). Loss = −min(ratio A, clip(ratio) A), with a value head and generalized advantage estimation. Reward can come from the RM or from a verifier."
	},
	{
		id: "T7",
		title: "GRPO is group-relative and has no value head",
		body: "Sample a group of answers, set advantage to (r − mean) / (std + ε) inside the group. That is the whole baseline. The repo treats this as RLVR: reinforcement learning from a verifiable reward, not a learned critic."
	},
	{
		id: "T8",
		title: "GSM8K greedy accuracy is the evaluation",
		body: "Generate greedily, parse the <answer> tag, compare to gold. eval_post_training.py prints one table across Base, SFT, DPO, PPO, and GRPO so the stages can be read as a single experiment."
	},
	{
		id: "T9",
		title: "Every orbit is handwritten PyTorch",
		body: "No trl, no peft, no transformers. The same modules that pretrain are the modules that SFT, reward, PPO, DPO, and GRPO. The flow is almost everywhere the same idea: change the data, change the loss, keep the net."
	}
];
var PIECES = [
	{
		id: "mlp",
		title: "MLP",
		file: "src/models/mlp.py",
		formula: "Linear(d, 4d) → ReLU → Linear(4d, d)",
		note: "The feed-forward expansion. One hidden layer, four times wide, then a projection back.",
		code: `class MLP(nn.Module):
    def __init__(self, n_embed):
        self.hidden = nn.Linear(n_embed, 4 * n_embed)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(4 * n_embed, n_embed)`
	},
	{
		id: "head",
		title: "Head",
		file: "src/models/attention.py",
		formula: "softmax(q kᵀ / √d · tril) v",
		note: "One causal head. Keys, queries, and values have no bias. The future is −∞ before softmax.",
		code: `attn = q @ k.transpose(-2, -1) * (1 / sqrt(d))
attn = attn.masked_fill(tril[:T, :T] == 0, -inf)
return softmax(attn, -1) @ v`
	},
	{
		id: "mha",
		title: "Multi-head",
		file: "src/models/attention.py",
		formula: "concat(h₁…hₙ) Wₒ",
		note: "n_head parallel Heads, concatenated, then a linear map back to n_embed.",
		code: `heads = [Head(n_embed // n_head, ...) for _ in range(n_head)]
x = torch.cat([h(x) for h in heads], dim=-1)
return self.proj(x)`
	},
	{
		id: "block",
		title: "Block",
		file: "src/models/transformer_block.py",
		formula: "x + attn(LN(x)); x + mlp(LN(x))",
		note: "Pre-norm residuals. Attention first, MLP second. This is the atom of depth.",
		code: `x = x + self.attn(self.ln1(x))
x = x + self.mlp(self.ln2(x))
return x`
	},
	{
		id: "tau",
		title: "Transformer",
		file: "src/models/transformer.py",
		formula: "Etok + Epos → Blockᴺ → LN → lm_head",
		note: "The whole net. generate() crops to the context window, softmaxes the last logit, and multinomials.",
		code: `x = token_embed(idx) + position_embed(pos)
for block in attn_blocks: x = block(x)
logits = lm_head(layer_norm(x))`
	}
];
function cn(...inputs) {
	return twMerge(clsx(inputs));
}
var METHODS = [
	{
		id: "SFT",
		title: "Supervised fine-tune",
		formula: "ce = CE(logits, y) * mask;  return ce.sum() / mask.sum()",
		note: "Instruction data (Alpaca, Dolly, GSM8K). Train only on assistant tokens. The base model learns to answer instead of to continue the internet."
	},
	{
		id: "RM",
		title: "Reward model",
		formula: "−log σ(r(chosen) − r(rejected))",
		note: "Bradley-Terry on preference pairs. Test accuracy in the README: 0.574 on 7 974 pairs. Weak, and enough to point PPO."
	},
	{
		id: "DPO",
		title: "DPO / ORPO / KTO",
		formula: "−log σ(β (π log-ratio − π_ref log-ratio))",
		note: "Preference without sampling from a reward model. ORPO drops the reference. KTO takes unpaired thumbs."
	},
	{
		id: "PPO",
		title: "PPO",
		formula: "−min(ratio A, clip(ratio) A)  with GAE",
		note: "Classic RLHF. Value head on the same backbone. Reward from the RM or from a verifier."
	},
	{
		id: "GRPO",
		title: "GRPO / RLVR",
		formula: "A = (r − mean_G) / (std_G + ε)",
		note: "Group of answers, relative advantage, no value head. The sink of the post-training line when the task can be checked."
	}
];
function AlignSection() {
	const [tid, setTid] = (0, import_react.useState)("T9");
	const theorem = THEOREMS.find((t) => t.id === tid) ?? THEOREMS[8];
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("section", {
		id: "align",
		className: "scroll-mt-16 border-b border-border",
		children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "mx-auto max-w-6xl px-4 py-16 sm:px-6 lg:py-20",
			children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "text-xs font-medium tracking-[0.22em] text-muted uppercase",
					children: "Post-training"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h2", {
					className: "font-display mt-3 text-4xl leading-tight font-medium tracking-[-0.03em] sm:text-5xl",
					children: "Turning a base into an assistant"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "mt-5 max-w-2xl text-base leading-relaxed text-muted",
					children: "After the sink of pretraining, the data changes and the loss changes. The net does not. Five methods, all handwritten, all on the same small Transformer."
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
					className: "mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3",
					children: METHODS.map((m) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("article", {
						className: "rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
								className: "font-mono text-xs tracking-[0.18em] text-subtle uppercase",
								children: m.id
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
								className: "font-display mt-3 text-2xl font-medium tracking-[-0.02em]",
								children: m.title
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
								className: "mt-3 font-mono text-xs leading-relaxed break-words text-fg sm:text-sm",
								children: m.formula
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
								className: "mt-3 text-sm leading-relaxed text-muted",
								children: m.note
							})
						]
					}, m.id))
				})
			]
		}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "mx-auto grid max-w-6xl gap-10 px-4 pb-16 sm:px-6 lg:grid-cols-[minmax(0,16rem)_1fr] lg:items-start lg:pb-20",
			children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "text-xs font-medium tracking-[0.22em] text-muted uppercase",
					children: "Theorems"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
					className: "font-display mt-3 text-3xl font-medium tracking-[-0.03em]",
					children: "T1–T9"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("ol", {
					className: "mt-6 space-y-1",
					children: THEOREMS.map((t) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("li", { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("button", {
						type: "button",
						onClick: () => setTid(t.id),
						className: cn("flex w-full items-baseline gap-3 rounded-md px-3 py-2 text-left text-sm transition-[background-color,color] duration-150 ease-out", tid === t.id ? "bg-elevated text-fg" : "text-muted hover:bg-elevated/60 hover:text-fg"),
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "w-8 shrink-0 font-mono text-xs text-mark",
							children: t.id
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", { children: t.title })]
					}) }, t.id))
				})
			] }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("article", {
				className: "rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6",
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "font-mono text-xs tracking-[0.18em] text-subtle uppercase",
						children: theorem.id
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h4", {
						className: "font-display mt-3 text-2xl font-medium tracking-[-0.02em]",
						children: theorem.title
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "mt-4 text-base leading-relaxed text-muted",
						children: theorem.body
					})
				]
			})]
		})]
	});
}
var PROBLEM = {
	id: "gsm8k-natalia",
	question: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
	gold: "72",
	reasoning: "April: 48. May: 48 / 2 = 24. Total: 48 + 24 = 72."
};
function parseAnswer(raw) {
	const tagged = raw.match(/<answer>\s*([^<]+?)\s*<\/answer>/i);
	if (tagged) return tagged[1].trim();
	const last = raw.trim().split(/\s+/).at(-1);
	if (!last) return null;
	return last.replace(/[^0-9.-]/g, "") || last;
}
function EvalSection() {
	const [draft, setDraft] = (0, import_react.useState)("<answer>72</answer>");
	const [submitted, setSubmitted] = (0, import_react.useState)(null);
	const parsed = (0, import_react.useMemo)(() => submitted === null ? null : parseAnswer(submitted), [submitted]);
	const correct = parsed !== null && parsed.replace(/,/g, "") === PROBLEM.gold;
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)("section", {
		id: "eval",
		className: "scroll-mt-16 border-b border-border",
		children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "mx-auto grid max-w-6xl gap-10 px-4 py-16 sm:px-6 lg:grid-cols-2 lg:items-start lg:gap-14 lg:py-20",
			children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "text-xs font-medium tracking-[0.22em] text-muted uppercase",
					children: "The evaluation"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h2", {
					className: "font-display mt-3 text-4xl leading-tight font-medium tracking-[-0.03em] sm:text-5xl",
					children: "GSM8K greedy"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
					className: "mt-5 text-base leading-relaxed text-muted",
					children: [
						"T8. Generate greedily, parse the",
						" ",
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "font-mono text-fg",
							children: "<answer>"
						}),
						" tag, compare to gold. That is the whole metric.",
						" ",
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "font-mono text-fg",
							children: "scripts/eval_post_training.py"
						}),
						" prints one table across Base, SFT, DPO, PPO, and GRPO so the stages can be read as a single experiment."
					]
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "mt-4 text-sm leading-relaxed text-subtle",
					children: "The box on the right is the parser, not a model. Type a completion, including a tag or a bare number, and see whether the sink accepts it."
				})
			] }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("form", {
				className: "rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6",
				onSubmit: (e) => {
					e.preventDefault();
					setSubmitted(draft);
				},
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "font-mono text-xs tracking-[0.18em] text-subtle uppercase",
						children: "C1 · Problem"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "mt-3 text-sm leading-relaxed text-fg",
						children: PROBLEM.question
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "mt-3 text-xs leading-relaxed text-subtle",
						children: PROBLEM.reasoning
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("label", {
						htmlFor: "completion",
						className: "mt-5 block text-sm font-medium text-fg",
						children: "Completion"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("textarea", {
						id: "completion",
						value: draft,
						onChange: (e) => setDraft(e.target.value),
						rows: 4,
						className: "mt-2 w-full resize-y rounded-md bg-elevated px-3 py-2.5 font-mono text-sm text-fg shadow-[var(--shadow-border)] outline-none focus-visible:ring-2 focus-visible:ring-ring/70"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "mt-4 flex flex-wrap gap-2",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
								type: "submit",
								className: "inline-flex h-11 items-center rounded-md bg-accent px-4 text-sm font-medium text-accent-fg transition-[scale,background-color] duration-150 ease-out hover:bg-fg active:scale-[0.96]",
								children: "Evaluate"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
								type: "button",
								onClick: () => {
									setDraft("<answer>72</answer>");
									setSubmitted(null);
								},
								className: "inline-flex h-11 items-center rounded-md px-4 text-sm font-medium text-fg shadow-[var(--shadow-border)] transition-[scale,background-color] duration-150 ease-out hover:bg-elevated/60 active:scale-[0.96]",
								children: "Gold"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
								type: "button",
								onClick: () => {
									setDraft("I think the total is 48.");
									setSubmitted(null);
								},
								className: "inline-flex h-11 items-center rounded-md px-4 text-sm font-medium text-fg shadow-[var(--shadow-border)] transition-[scale,background-color] duration-150 ease-out hover:bg-elevated/60 active:scale-[0.96]",
								children: "Miss"
							})
						]
					}),
					submitted !== null ? /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("dl", {
						className: "mt-5 space-y-2 border-t border-border pt-5 text-sm",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "flex justify-between gap-4",
								children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("dt", {
									className: "text-muted",
									children: "Parsed"
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("dd", {
									className: "font-mono text-fg",
									children: parsed ?? "—"
								})]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "flex justify-between gap-4",
								children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("dt", {
									className: "text-muted",
									children: "Gold"
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("dd", {
									className: "font-mono text-fg",
									children: PROBLEM.gold
								})]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "flex justify-between gap-4",
								children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("dt", {
									className: "text-muted",
									children: "Verdict"
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("dd", {
									className: cn("font-mono", correct ? "text-fg" : "text-mark"),
									children: correct ? "match" : "miss"
								})]
							})
						]
					}) : null
				]
			})]
		})
	});
}
function SiteFooter() {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)("footer", {
		className: "border-t border-border",
		children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "mx-auto flex max-w-6xl flex-col gap-8 px-4 py-10 sm:flex-row sm:items-end sm:justify-between sm:px-6",
			children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "font-display text-2xl italic text-fg",
					children: "τ"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "mt-2 max-w-md text-sm leading-relaxed text-muted",
					children: "Training an LLM from scratch — an interactive reading of Fareed Khan’s repository. A defined path code can certify, and a name fitted rather than free."
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
					className: "mt-4 max-w-md text-sm leading-relaxed text-muted",
					children: [
						"Source:",
						" ",
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
							href: REPO,
							target: "_blank",
							rel: "noopener noreferrer",
							className: "inline-flex min-h-11 items-center text-fg underline decoration-border underline-offset-4 transition-[color,text-decoration-color] duration-150 ease-out hover:decoration-fg",
							children: "train-llm-from-scratch"
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "text-subtle",
							children: " · "
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
							href: DOCS,
							target: "_blank",
							rel: "noopener noreferrer",
							className: "inline-flex min-h-11 items-center text-fg underline decoration-border underline-offset-4 transition-[color,text-decoration-color] duration-150 ease-out hover:decoration-fg",
							children: "Docs"
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "text-subtle",
							children: " · "
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
							href: PAPER,
							target: "_blank",
							rel: "noopener noreferrer",
							className: "inline-flex min-h-11 items-center text-fg underline decoration-border underline-offset-4 transition-[color,text-decoration-color] duration-150 ease-out hover:decoration-fg",
							children: "Attention is All You Need"
						})
					]
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
					className: "mt-1 text-sm leading-relaxed text-muted",
					children: [
						"Author:",
						" ",
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
							href: AUTHOR,
							target: "_blank",
							rel: "noopener noreferrer",
							className: "inline-flex min-h-11 items-center text-fg underline decoration-border underline-offset-4 transition-[color,text-decoration-color] duration-150 ease-out hover:decoration-fg",
							children: "Fareed Khan"
						})
					]
				})
			] }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
				className: "text-xs tracking-wide text-subtle",
				children: "Form, not product. Occupied, not wrapped."
			})]
		})
	});
}
var LINKS = [
	{
		href: "#path",
		label: "Path"
	},
	{
		href: "#model",
		label: "Model"
	},
	{
		href: "#loss",
		label: "Loss"
	},
	{
		href: "#align",
		label: "Align"
	},
	{
		href: "#eval",
		label: "Eval"
	},
	{
		href: "#name",
		label: "Name"
	},
	{
		href: "#repo",
		label: "Repo"
	}
];
function SiteHeader() {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)("header", {
		className: "sticky top-0 z-40 border-b border-border bg-bg/85 backdrop-blur-md",
		children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "mx-auto flex h-14 max-w-6xl min-w-0 items-center justify-between gap-3 px-4 sm:h-16 sm:gap-4 sm:px-6",
			children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("a", {
				href: "#top",
				className: "flex min-w-0 shrink items-baseline gap-2 no-underline",
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
					className: "font-display text-2xl leading-none text-fg italic",
					children: "τ"
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
					className: "truncate text-xs font-medium tracking-[0.14em] text-fg uppercase sm:text-sm sm:tracking-[0.16em]",
					children: "From Scratch"
				})]
			}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("nav", {
				"aria-label": "Sections",
				className: "flex min-w-0 flex-1 items-center justify-end gap-1 overflow-x-auto sm:gap-0",
				children: LINKS.map((link) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
					href: link.href,
					className: "shrink-0 rounded-md px-2.5 py-2 text-xs font-medium tracking-wide text-muted no-underline transition-[color,background-color] duration-150 ease-out hover:text-fg focus-visible:ring-2 focus-visible:ring-ring/70 sm:px-3 sm:text-sm",
					children: link.label
				}, link.href))
			})]
		})
	});
}
function Hero() {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("section", {
		id: "top",
		className: "relative overflow-hidden border-b border-border",
		children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
			"aria-hidden": "true",
			className: "tau-watermark absolute -right-6 top-10 text-fg/5 sm:right-10 sm:top-6",
			children: "τ"
		}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "stagger-in mx-auto flex max-w-6xl flex-col gap-8 px-4 py-16 sm:px-6 sm:py-24 lg:py-28",
			children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
					className: "flex items-baseline gap-3 text-xs font-medium tracking-[0.22em] text-muted uppercase",
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
						className: "font-display text-2xl leading-none font-medium tracking-normal text-fg italic normal-case",
						children: "τ"
					}), "After Attention, without the wrappers"]
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("h1", {
					className: "font-display max-w-3xl text-5xl leading-[0.92] font-medium tracking-[-0.03em] text-fg sm:text-6xl lg:text-7xl",
					children: ["Training an LLM from ", /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
						className: "italic",
						children: "scratch"
					})]
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "max-w-xl text-base leading-relaxed text-muted sm:text-lg",
					children: "Code proves a path, not a product. Here the path is occupied: one architecture, a first loss, a source-line of stages, and a unique sink at chat. From-scratch names that path — fitted, not forced."
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "flex flex-wrap items-center gap-3",
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
						href: "#path",
						className: "inline-flex h-11 items-center rounded-md bg-accent px-4 text-sm font-medium text-accent-fg no-underline transition-[scale,background-color] duration-150 ease-out hover:bg-fg active:scale-[0.96]",
						children: "Walk the source-line"
					}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
						href: "#model",
						className: "inline-flex h-11 items-center rounded-md px-4 text-sm font-medium text-fg no-underline shadow-[var(--shadow-border)] transition-[scale,box-shadow,background-color] duration-150 ease-out hover:bg-elevated/60 hover:shadow-[var(--shadow-border-hover)] active:scale-[0.96]",
						children: "Compute τ(d)"
					})]
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "flex flex-wrap items-baseline gap-x-1 gap-y-1 font-mono text-xs leading-relaxed tracking-wide text-subtle sm:text-sm",
					children: PIPELINE.map((step, i) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", { children: [
						i > 0 ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "text-subtle/50",
							children: " → "
						}) : null,
						step.label,
						step.sink ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)("sup", {
							className: "text-mark",
							children: "∞"
						}) : null
					] }, step.id))
				})
			]
		})]
	});
}
var BASE_CORPUS = `In 1978, The park was returned to the factory-plate that
the public share to the lower of the electronic fence that
follow from the Station's cities. The Canal of ancient Western
nations were confined to the city spot. The villages were directly
linked to cities in China that revolt that the US budget and in
Odambinais is uncertain and fortune established in rural areas.
The transformer was returned to the factory of language that
the public share to the lower of the electronic fence.
Ancient Western nations were confined to the city spot and
the villages were directly linked to cities that revolt.
`;
function train(corpus, order) {
	const table = /* @__PURE__ */ new Map();
	const padded = corpus.replace(/\s+/g, " ").trim();
	for (let i = 0; i <= padded.length - order; i++) {
		const key = padded.slice(i, i + order);
		const next = padded[i + order] ?? "";
		const arr = table.get(key);
		if (arr) arr.push(next);
		else table.set(key, [next]);
	}
	return table;
}
var ORDER = 3;
var BASE_TABLE = train(BASE_CORPUS, ORDER);
function sampleFrom(table, prefix, maxChars, temperature) {
	let out = prefix;
	let guard = 0;
	while (out.length < prefix.length + maxChars && guard++ < 800) {
		const key = out.slice(-3);
		const options = table.get(key);
		if (!options || options.length === 0) break;
		const pick = options[Math.floor(Math.random() * options.length)] ?? "";
		if (!pick) break;
		if (temperature < .4 && options.length > 2) {
			const counts = /* @__PURE__ */ new Map();
			for (const ch of options) counts.set(ch, (counts.get(ch) ?? 0) + 1);
			let best = options[0];
			let bestN = 0;
			for (const [ch, n] of counts) if (n > bestN) {
				best = ch;
				bestN = n;
			}
			out += best;
		} else out += pick;
	}
	return out;
}
function generateBase(prompt, maxChars = 220, temperature = .9) {
	const seed = (prompt.trim() || "The ").slice(0, 80);
	let start = seed;
	if (start.length < ORDER) start = (BASE_CORPUS.slice(0, ORDER) + start).slice(0, ORDER);
	return (seed + sampleFrom(BASE_TABLE, start.slice(-3), maxChars, temperature).slice(ORDER)).replace(/\s+/g, " ").trim();
}
var STAGE_REPLIES = {
	base: [
		{
			prompt: "What is 2 + 2?",
			reply: "In 1978, The park was returned to the factory-plate that the public share to the lower of the electronic fence that follow from the Station's cities."
		},
		{
			prompt: "Explain attention.",
			reply: "The Canal of ancient Western nations were confined to the city spot. The villages were directly linked to cities in China that revolt that the US budget."
		},
		{
			prompt: "A park in 1978.",
			reply: "In 1978, The park was returned to the factory-plate that the public share to the lower of the electronic fence that follow from the Station's cities."
		}
	],
	sft: [
		{
			prompt: "What is 2 + 2?",
			reply: "2 + 2 equals 4."
		},
		{
			prompt: "Explain attention.",
			reply: "Attention lets each token look at the others and decide which ones matter. In a causal language model, a token may only look at itself and what came before."
		},
		{
			prompt: "A park in 1978.",
			reply: "Write a short paragraph about a park that was returned to the public in 1978, with a fence, a canal, and nearby villages."
		}
	],
	grpo: [
		{
			prompt: "What is 2 + 2?",
			reply: "Adding two and two gives four.\n<answer>4</answer>"
		},
		{
			prompt: "Explain attention.",
			reply: "Causal self-attention computes q kᵀ / √d, masks the future, softmaxes, and mixes values. That is the whole move.\n<answer>causal scaled-dot-product attention</answer>"
		},
		{
			prompt: "A park in 1978.",
			reply: "A public park restored in 1978, bounded by a fence and a canal, with villages linked to the city.\n<answer>1978 public park</answer>"
		}
	]
};
var POINTS = [
	{
		t: 0,
		train: 11.14,
		dev: 11.2
	},
	{
		t: 250,
		train: 8.4,
		dev: 8.55
	},
	{
		t: 500,
		train: 6.35,
		dev: 6.5
	},
	{
		t: 1e3,
		train: 4.82,
		dev: 4.95
	},
	{
		t: 1500,
		train: 4.12,
		dev: 4.22
	},
	{
		t: 2e3,
		train: 3.73,
		dev: 3.76
	}
];
function lerpLoss(step) {
	const first = POINTS[0];
	const last = POINTS[POINTS.length - 1];
	if (step <= 0) return first;
	if (step >= 2e3) return last;
	let a = first;
	let b = last;
	for (let i = 0; i < POINTS.length - 1; i++) {
		const lo = POINTS[i];
		const hi = POINTS[i + 1];
		if (step >= lo.t && step <= hi.t) {
			a = lo;
			b = hi;
			break;
		}
	}
	const u = (step - a.t) / (b.t - a.t || 1);
	return {
		t: step,
		train: a.train + (b.train - a.train) * u,
		dev: a.dev + (b.dev - a.dev) * u
	};
}
function LossSection() {
	const [step, setStep] = (0, import_react.useState)(2e3);
	const [stage, setStage] = (0, import_react.useState)("base");
	const [promptIdx, setPromptIdx] = (0, import_react.useState)(0);
	const [generated, setGenerated] = (0, import_react.useState)(null);
	const loss = (0, import_react.useMemo)(() => lerpLoss(step), [step]);
	const replies = STAGE_REPLIES[stage];
	const pair = replies[promptIdx] ?? replies[0];
	function runGenerate() {
		if (stage === "base") setGenerated(generateBase(pair.prompt, 200, .95));
		else setGenerated(pair.reply);
	}
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("section", {
		id: "loss",
		className: "scroll-mt-16 border-b border-border",
		children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "mx-auto grid max-w-6xl gap-10 px-4 py-16 sm:px-6 lg:grid-cols-2 lg:items-start lg:gap-14 lg:py-20",
			children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "text-xs font-medium tracking-[0.22em] text-muted uppercase",
					children: "Unique sink of pretraining"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h2", {
					className: "font-display mt-3 text-4xl leading-tight font-medium tracking-[-0.03em] sm:text-5xl",
					children: "Next-token loss"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "mt-5 text-base leading-relaxed text-muted",
					children: "Every orbit of raw text reaches the same objective. Write the next token, grade the logits, descend. The 77 million parameter run in the README is the exhibited orbit: 11.14 → 3.73 train, 3.76 on the held-out slice, in 2 000 steps."
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
					className: "mt-4 font-mono text-sm break-words text-fg",
					children: [
						"L = CE(softmax(z), y",
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("sub", { children: "t+1" }),
						")"
					]
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "mt-4 text-sm leading-relaxed text-subtle",
					children: "T2. Independently checked in the training logs. The curve here is a reading of those two endpoints, not a re-run."
				})
			] }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6",
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "flex items-baseline justify-between gap-3",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
							className: "font-mono text-xs tracking-[0.18em] text-subtle uppercase",
							children: "Orbit · 77M"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
							className: "font-mono text-xs tabular-nums text-muted",
							children: ["step ", step.toLocaleString("en-US")]
						})]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "font-display mt-4 text-5xl leading-none font-medium tracking-[-0.04em] tabular-nums text-fg",
						children: loss.train.toFixed(2)
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
						className: "mt-2 text-sm text-muted",
						children: ["train · dev ", loss.dev.toFixed(2)]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("label", {
						className: "mt-6 block text-sm font-medium text-fg",
						htmlFor: "step-slider",
						children: "Training step"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("input", {
						id: "step-slider",
						type: "range",
						min: 0,
						max: 2e3,
						step: 10,
						value: step,
						onChange: (e) => setStep(Number(e.target.value)),
						className: "mt-2 w-full accent-[var(--color-mark)]"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "mt-5 space-y-3",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							className: "mb-1 flex justify-between text-xs text-subtle",
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", { children: "train" }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "font-mono tabular-nums",
								children: loss.train.toFixed(2)
							})]
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "loss-bar",
							children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", { style: { width: `${loss.train / 11.2 * 100}%` } })
						})] }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							className: "mb-1 flex justify-between text-xs text-subtle",
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", { children: "dev" }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "font-mono tabular-nums",
								children: loss.dev.toFixed(2)
							})]
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
							className: "loss-bar",
							children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", { style: { width: `${loss.dev / 11.2 * 100}%` } })
						})] })]
					})
				]
			})]
		}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "mx-auto max-w-6xl px-4 pb-16 sm:px-6 lg:pb-20",
			children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "text-xs font-medium tracking-[0.22em] text-muted uppercase",
					children: "Where the small end starts"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
					className: "font-display mt-3 text-3xl font-medium tracking-[-0.03em]",
					children: "Generate"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "mt-4 max-w-2xl text-sm leading-relaxed text-muted",
					children: "Below is the output of a trained 13 million parameter LLM, just so you can see where the small end of this starts — and how SFT then GRPO change the mouth of the same net. Base generation is a character Markov reading of that sample. Later stages are staged replies, not a hosted checkpoint."
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
					className: "mt-6 flex flex-wrap gap-2",
					children: [
						"base",
						"sft",
						"grpo"
					].map((s) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
						type: "button",
						onClick: () => {
							setStage(s);
							setGenerated(null);
						},
						className: cn("h-10 rounded-md px-3 font-mono text-sm uppercase tracking-wide shadow-[var(--shadow-border)] transition-[background-color,color] duration-150 ease-out", stage === s ? "bg-elevated text-fg" : "text-muted hover:bg-elevated hover:text-fg"),
						children: s
					}, s))
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
					className: "mt-4 flex flex-wrap gap-2",
					children: STAGE_REPLIES.base.map((p, i) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
						type: "button",
						onClick: () => {
							setPromptIdx(i);
							setGenerated(null);
						},
						className: cn("h-10 rounded-md px-3 text-sm shadow-[var(--shadow-border)] transition-[background-color,color] duration-150 ease-out", promptIdx === i ? "bg-elevated text-fg" : "text-muted hover:bg-elevated hover:text-fg"),
						children: p.prompt
					}, p.prompt))
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "mt-5 grid gap-4 lg:grid-cols-[1fr_auto] lg:items-start",
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("article", {
						className: "rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
								className: "font-mono text-xs tracking-[0.18em] text-subtle uppercase",
								children: [stage, " · prompt"]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
								className: "mt-3 text-sm text-fg",
								children: pair.prompt
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("pre", {
								className: "mt-4 whitespace-pre-wrap font-mono text-sm leading-relaxed text-muted",
								children: generated ?? pair.reply
							})
						]
					}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
						type: "button",
						onClick: runGenerate,
						className: "inline-flex h-11 items-center justify-center rounded-md bg-accent px-4 text-sm font-medium text-accent-fg transition-[scale,background-color] duration-150 ease-out hover:bg-fg active:scale-[0.96]",
						children: "Sample again"
					})]
				})
			]
		})]
	});
}
var VOCAB_SIZE = 50304;
var PRESETS = [
	{
		id: "13m",
		label: "13M",
		nEmbed: 128,
		nHead: 8,
		nBlocks: 1,
		contextLength: 256,
		reported: 13142656
	},
	{
		id: "77m",
		label: "77M",
		nEmbed: 512,
		nHead: 8,
		nBlocks: 8,
		contextLength: 512,
		reported: 77031552
	},
	{
		id: "406m",
		label: "406M",
		nEmbed: 1024,
		nHead: 16,
		nBlocks: 24,
		contextLength: 1024,
		reported: 406359168
	}
];
/** Exact nn.Parameter count of the handwritten Transformer in the repo. */
function countParams(spec) {
	const { nEmbed, nHead, nBlocks, contextLength, vocabSize } = spec;
	const headSize = Math.floor(nEmbed / nHead);
	const attn = nHead * (3 * nEmbed * headSize) + nEmbed * nEmbed + nEmbed;
	const mlp = nEmbed * (4 * nEmbed) + 4 * nEmbed + 4 * nEmbed * nEmbed + nEmbed;
	const ln = 4 * nEmbed;
	const block = attn + mlp + ln;
	const token = vocabSize * nEmbed;
	const pos = contextLength * nEmbed;
	const finalLn = 2 * nEmbed;
	const lmHead = nEmbed * vocabSize + vocabSize;
	return {
		token,
		pos,
		attn: nBlocks * attn,
		mlp: nBlocks * mlp,
		ln: nBlocks * ln + finalLn,
		lmHead,
		block,
		total: token + pos + nBlocks * block + finalLn + lmHead,
		headSize
	};
}
function formatInt(n) {
	const rounded = Math.round(n);
	return (rounded < 0 ? "−" : "") + Math.abs(rounded).toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
}
function formatCompact(n) {
	if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
	if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)}M`;
	if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
	return formatInt(n);
}
var GPUS = [
	{
		name: "NVIDIA A100",
		memory: "40 GB",
		max: "~6B–8B",
		fit13: true,
		fit2b: true
	},
	{
		name: "NVIDIA V100",
		memory: "16 GB",
		max: "~2B",
		fit13: true,
		fit2b: false
	},
	{
		name: "NVIDIA RTX 4090",
		memory: "24 GB",
		max: "~4B",
		fit13: true,
		fit2b: true
	},
	{
		name: "NVIDIA RTX 5090",
		memory: "32 GB",
		max: "13M verified",
		fit13: true,
		fit2b: true
	},
	{
		name: "NVIDIA RTX 3090",
		memory: "24 GB",
		max: "~3.5B–4B",
		fit13: true,
		fit2b: true
	},
	{
		name: "NVIDIA RTX 4080",
		memory: "16 GB",
		max: "~2B",
		fit13: true,
		fit2b: false
	},
	{
		name: "NVIDIA RTX 4060",
		memory: "8 GB",
		max: "~1B",
		fit13: true,
		fit2b: false
	},
	{
		name: "Tesla T4",
		memory: "16 GB",
		max: "~1.5B–2B",
		fit13: true,
		fit2b: false
	}
];
function ModelSection() {
	const [presetId, setPresetId] = (0, import_react.useState)("13m");
	const [nEmbed, setNEmbed] = (0, import_react.useState)(128);
	const [nHead, setNHead] = (0, import_react.useState)(8);
	const [nBlocks, setNBlocks] = (0, import_react.useState)(1);
	const [context, setContext] = (0, import_react.useState)(256);
	const [pieceId, setPieceId] = (0, import_react.useState)("block");
	const preset = PRESETS.find((p) => p.id === presetId);
	const spec = {
		nEmbed,
		nHead,
		nBlocks,
		contextLength: context,
		vocabSize: VOCAB_SIZE
	};
	const broken = nHead < 1 || nEmbed % nHead !== 0;
	const counted = (0, import_react.useMemo)(() => broken ? null : countParams(spec), [
		broken,
		nEmbed,
		nHead,
		nBlocks,
		context
	]);
	const displayTotal = preset ? preset.reported : counted?.total ?? 0;
	const piece = PIECES.find((p) => p.id === pieceId) ?? PIECES[3];
	function applyPreset(id) {
		const p = PRESETS.find((x) => x.id === id);
		setPresetId(id);
		setNEmbed(p.nEmbed);
		setNHead(p.nHead);
		setNBlocks(p.nBlocks);
		setContext(p.contextLength);
	}
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("section", {
		id: "model",
		className: "scroll-mt-16 border-b border-border",
		children: [
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "mx-auto grid max-w-6xl gap-10 px-4 py-16 sm:px-6 lg:grid-cols-2 lg:items-start lg:gap-14 lg:py-20",
				children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "flex min-w-0 flex-col gap-5",
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
							className: "text-xs font-medium tracking-[0.22em] text-muted uppercase",
							children: "The architecture"
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h2", {
							className: "font-display text-4xl leading-tight font-medium tracking-[-0.03em] sm:text-5xl",
							children: "Analeza, for nets"
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
							className: "text-base leading-relaxed text-muted",
							children: "τ is the kucwenga map of this repo: write a model as (embed × 1), split every block down to MLP and Head, then sum every Linear the splits produce. For n_embed = d the lemma gives a single formula, independent of split order."
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
							className: "font-mono text-sm break-words text-fg",
							children: "τ(d, N, T, V) = Vd + Td + N(12d² + 10d) + 2d + Vd + V"
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
							className: "text-xs leading-relaxed text-subtle",
							children: "Vocab V is 50 304. Untied lm_head. Attention keys have no bias; MLP and projections do. Reported 13M / 77M / 406M counts come from the README; custom uses the formula above."
						})
					]
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("form", {
					className: "rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6",
					onSubmit: (e) => e.preventDefault(),
					children: [
						/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
							className: "font-mono text-xs tracking-[0.18em] text-subtle uppercase",
							children: "Evaluate τ"
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							className: "mt-4 flex flex-wrap gap-2",
							children: [PRESETS.map((p) => /* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
								type: "button",
								onClick: () => applyPreset(p.id),
								className: cn("h-10 min-w-10 rounded-md px-3 font-mono text-sm tabular-nums shadow-[var(--shadow-border)] transition-[background-color,box-shadow,color] duration-150 ease-out", presetId === p.id ? "bg-elevated text-fg" : "text-muted hover:bg-elevated hover:text-fg"),
								children: p.label
							}, p.id)), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
								type: "button",
								onClick: () => setPresetId("custom"),
								className: cn("h-10 min-w-10 rounded-md px-3 font-mono text-sm shadow-[var(--shadow-border)] transition-[background-color,box-shadow,color] duration-150 ease-out", presetId === "custom" ? "bg-elevated text-fg" : "text-muted hover:bg-elevated hover:text-fg"),
								children: "custom"
							})]
						}),
						/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
							className: "mt-5 grid gap-4 sm:grid-cols-2",
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(NumberField, {
									label: "n_embed d",
									value: nEmbed,
									min: 64,
									max: 2048,
									step: 64,
									onChange: (v) => {
										setPresetId("custom");
										setNEmbed(v);
									}
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(NumberField, {
									label: "n_head",
									value: nHead,
									min: 1,
									max: 32,
									step: 1,
									onChange: (v) => {
										setPresetId("custom");
										setNHead(v);
									}
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(NumberField, {
									label: "N_BLOCKS",
									value: nBlocks,
									min: 1,
									max: 48,
									step: 1,
									onChange: (v) => {
										setPresetId("custom");
										setNBlocks(v);
									}
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsx)(NumberField, {
									label: "context T",
									value: context,
									min: 128,
									max: 2048,
									step: 128,
									onChange: (v) => {
										setPresetId("custom");
										setContext(v);
									}
								})
							]
						}),
						broken ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
							className: "mt-4 text-sm text-mark",
							children: "d must be divisible by n_head. Head size is d / n_head."
						}) : /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("dl", {
							className: "mt-5 space-y-2 border-t border-border pt-5 text-sm",
							children: [
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
									className: "flex justify-between gap-4",
									children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("dt", {
										className: "text-muted",
										children: "Parameters"
									}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("dd", {
										className: "font-mono tabular-nums text-fg",
										children: formatInt(displayTotal)
									})]
								}),
								/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
									className: "flex justify-between gap-4",
									children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("dt", {
										className: "text-muted",
										children: "Compact"
									}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("dd", {
										className: "font-mono tabular-nums text-fg",
										children: formatCompact(displayTotal)
									})]
								}),
								counted ? /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
									className: "flex justify-between gap-4",
									children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("dt", {
										className: "text-muted",
										children: "Head size"
									}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("dd", {
										className: "font-mono tabular-nums text-fg",
										children: counted.headSize
									})]
								}) : null
							]
						})
					]
				})]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "mx-auto max-w-6xl px-4 pb-10 sm:px-6",
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "text-xs font-medium tracking-[0.22em] text-muted uppercase",
						children: "The chamber"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
						className: "font-display mt-3 text-3xl font-medium tracking-[-0.03em] sm:text-4xl",
						children: "Five seats, not fourteen"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
						className: "mt-4 max-w-2xl text-sm leading-relaxed text-muted",
						children: [
							"L_τ = ",
							"{",
							"MLP, Head, Multi-head, Block, Transformer",
							"}",
							". Five — not chosen to match a paper, only noticed after the files were named. Click a seat."
						]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						className: "mt-6 grid gap-2 sm:grid-cols-5",
						children: PIECES.map((p) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("button", {
							type: "button",
							onClick: () => setPieceId(p.id),
							className: cn("rounded-lg px-3 py-3 text-left shadow-[var(--shadow-border)] transition-[background-color,box-shadow,color] duration-150 ease-out", pieceId === p.id ? "bg-elevated text-fg" : "bg-surface text-muted hover:text-fg"),
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "font-display text-xl italic",
								children: p.title
							}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "mt-1 block font-mono text-[11px] tracking-wide text-subtle",
								children: p.file
							})]
						}, p.id))
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("article", {
						className: "mt-4 rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
								className: "font-mono text-sm break-words text-fg",
								children: piece.formula
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
								className: "mt-3 text-sm leading-relaxed text-muted",
								children: piece.note
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("pre", {
								className: "mt-4 overflow-x-auto rounded-lg bg-elevated p-4 font-mono text-xs leading-relaxed text-accent shadow-[var(--shadow-border)]",
								children: piece.code
							})
						]
					})
				]
			}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "mx-auto max-w-6xl px-4 pb-16 sm:px-6 lg:pb-20",
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "text-xs font-medium tracking-[0.22em] text-muted uppercase",
						children: "Hardware"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
						className: "font-display mt-3 text-2xl font-medium tracking-[-0.02em]",
						children: "Which GPUs hold τ"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						className: "mt-5 max-w-full overflow-x-auto",
						children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("table", {
							className: "w-full min-w-[36rem] text-left text-sm",
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("thead", { children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("tr", {
								className: "border-b border-border text-xs tracking-[0.14em] text-subtle uppercase",
								children: [
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", {
										className: "py-2 pr-4 font-medium",
										children: "GPU"
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", {
										className: "py-2 pr-4 font-medium",
										children: "Memory"
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", {
										className: "py-2 pr-4 font-medium",
										children: "13M"
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", {
										className: "py-2 pr-4 font-medium",
										children: "~2B"
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("th", {
										className: "py-2 font-medium",
										children: "Practical max"
									})
								]
							}) }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("tbody", { children: GPUS.map((g) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("tr", {
								className: "border-b border-border/80",
								children: [
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", {
										className: "py-2.5 pr-4 text-fg",
										children: g.name
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", {
										className: "py-2.5 pr-4 font-mono tabular-nums text-muted",
										children: g.memory
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", {
										className: "py-2.5 pr-4 font-mono text-mark",
										children: g.fit13 ? "✔" : "✘"
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", {
										className: "py-2.5 pr-4 font-mono text-muted",
										children: g.fit2b ? "✔" : "✘"
									}),
									/* @__PURE__ */ (0, import_jsx_runtime.jsx)("td", {
										className: "py-2.5 font-mono text-muted",
										children: g.max
									})
								]
							}, g.name)) })]
						})
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
						className: "mt-4 text-xs leading-relaxed text-subtle",
						children: [
							"A free Colab or Kaggle T4 is enough for 13 million parameters. A billion-parameter run is not. Flags ",
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "text-fg",
								children: "--amp"
							}),
							",",
							" ",
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "text-fg",
								children: "--grad-checkpointing"
							}),
							",",
							" ",
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "text-fg",
								children: "--grad-accum"
							}),
							" bring the memory down when a large config refuses."
						]
					})
				]
			})
		]
	});
}
function NumberField({ label, value, min, max, step, onChange }) {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("label", {
		className: "flex flex-col gap-1.5 text-sm font-medium text-fg",
		children: [label, /* @__PURE__ */ (0, import_jsx_runtime.jsx)("input", {
			type: "number",
			inputMode: "numeric",
			min,
			max,
			step,
			value,
			onChange: (e) => {
				const v = Number(e.target.value);
				if (Number.isFinite(v)) onChange(v);
			},
			className: "h-11 w-full rounded-md bg-elevated px-3 font-mono text-base text-fg tabular-nums shadow-[var(--shadow-border)] outline-none focus-visible:ring-2 focus-visible:ring-ring/70"
		})]
	});
}
function NameSection() {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)("section", {
		id: "name",
		className: "scroll-mt-16 border-b border-border",
		children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "mx-auto grid max-w-6xl gap-10 px-4 py-16 sm:px-6 lg:grid-cols-2 lg:items-start lg:gap-16 lg:py-20",
			children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "text-xs font-medium tracking-[0.22em] text-muted uppercase",
					children: "Fit, not force"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h2", {
					className: "font-display mt-3 text-4xl leading-tight font-medium tracking-[-0.03em] sm:text-5xl",
					children: "The name"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "mt-5 text-base leading-relaxed text-muted",
					children: "A substitution test shows that T1–T9 do not print the phrase “from scratch.” Replace it with any wrapper — Hugging Face transformers, trl, peft — and every sum of the handwritten modules still compiles. That test is valid and small. It shows only that arithmetic of tensors does not output a title."
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "mt-4 text-base leading-relaxed text-muted",
					children: "It does not show that every title fits τ equally. “From scratch,” in the inherited sense, already carries content: one net, built of small files, trained on public data, aligned by losses you can read. τ exhibits a single architecture, a first triad of stages, and a unique sink for what those stages generate."
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "mt-4 text-base leading-relaxed text-muted",
					children: "This is a semantic argument, not a tenth evaluation of CE. It stands or falls on whether MLP, Head, and Block carry the structural role claimed for them. The README should not present it as a theorem. It is the name of the path."
				})
			] }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("aside", {
				className: "mt-4 rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6 lg:mt-12",
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
						className: "font-display text-2xl leading-snug font-medium tracking-[-0.02em] italic text-fg",
						children: [
							"τ = (MLP, Head, Blockᴺ, CE, SFT → ",
							"{",
							"DPO, PPO, GRPO",
							"}",
							")"
						]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "mt-5 text-sm leading-relaxed text-muted",
						children: "From-scratch, in this reading, names τ. Existence as a hosted billion-parameter service is not a line of this map. Existence as form is."
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("dl", {
						className: "mt-6 space-y-3 border-t border-border pt-5 text-sm",
						children: [
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "flex gap-4",
								children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("dt", {
									className: "w-24 shrink-0 font-medium text-fg",
									children: "Proved"
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("dd", {
									className: "text-muted",
									children: "T1–T9, the formula, the CE barrier, unique sink at chat"
								})]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "flex gap-4",
								children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("dt", {
									className: "w-24 shrink-0 font-medium text-fg",
									children: "Stipulated"
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("dd", {
									className: "text-muted",
									children: "r50k_base, ReLU not GELU, untied lm_head, pre-norm"
								})]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "flex gap-4",
								children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("dt", {
									className: "w-24 shrink-0 font-medium text-fg",
									children: "Fit"
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("dd", {
									className: "text-muted",
									children: "The name from-scratch, pedagogical"
								})]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "flex gap-4",
								children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("dt", {
									className: "w-24 shrink-0 font-medium text-fg",
									children: "Evidence"
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("dd", {
									className: "text-muted",
									children: "13M / 77M / 406M; GSM8K table; RM accuracy 0.574; Attention is All You Need"
								})]
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
								className: "flex gap-4",
								children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("dt", {
									className: "w-24 shrink-0 font-medium text-fg",
									children: "Refused"
								}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("dd", {
									className: "text-muted",
									children: "transformers, trl, peft; proof of a product"
								})]
							})
						]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
						className: "mt-6 text-sm leading-relaxed text-muted",
						children: [
							"Source:",
							" ",
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
								href: REPO,
								target: "_blank",
								rel: "noopener noreferrer",
								className: "text-fg underline decoration-border underline-offset-4 transition-[color,text-decoration-color] duration-150 ease-out hover:decoration-fg",
								children: "FareedKhan-dev/train-llm-from-scratch"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "text-subtle",
								children: " · "
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
								href: DOCS,
								target: "_blank",
								rel: "noopener noreferrer",
								className: "text-fg underline decoration-border underline-offset-4 transition-[color,text-decoration-color] duration-150 ease-out hover:decoration-fg",
								children: "Docs"
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "text-subtle",
								children: " · "
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
								href: PAPER,
								target: "_blank",
								rel: "noopener noreferrer",
								className: "text-fg underline decoration-border underline-offset-4 transition-[color,text-decoration-color] duration-150 ease-out hover:decoration-fg",
								children: "Vaswani et al."
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "text-subtle",
								children: " · "
							}),
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
								href: AUTHOR,
								target: "_blank",
								rel: "noopener noreferrer",
								className: "text-fg underline decoration-border underline-offset-4 transition-[color,text-decoration-color] duration-150 ease-out hover:decoration-fg",
								children: "Fareed Khan"
							})
						]
					})
				]
			})]
		})
	});
}
var GPT2_RE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
function splitGpt2Pieces(text) {
	if (!text) return [];
	return text.match(GPT2_RE) ?? [];
}
function standInTokenId(piece) {
	let h = 2166136261;
	for (let i = 0; i < piece.length; i++) {
		h ^= piece.charCodeAt(i);
		h = Math.imul(h, 16777619);
	}
	return (h >>> 0) % 50257;
}
function tokenize(text) {
	return splitGpt2Pieces(text).map((piece) => ({
		piece,
		id: standInTokenId(piece)
	}));
}
var DEFAULT_TEXT = "Attention is all you need to train an LLM from scratch.";
function PathSection() {
	const [text, setText] = (0, import_react.useState)(DEFAULT_TEXT);
	const [cursor, setCursor] = (0, import_react.useState)(0);
	const [playing, setPlaying] = (0, import_react.useState)(false);
	const tokens = (0, import_react.useMemo)(() => tokenize(text), [text]);
	(0, import_react.useEffect)(() => {
		if (!playing) return;
		const id = window.setInterval(() => {
			setCursor((c) => {
				if (c >= PIPELINE.length - 1) {
					setPlaying(false);
					return c;
				}
				return c + 1;
			});
		}, 900);
		return () => window.clearInterval(id);
	}, [playing]);
	const current = PIPELINE[cursor] ?? PIPELINE[0];
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("section", {
		id: "path",
		className: "scroll-mt-16 border-b border-border",
		children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "mx-auto grid max-w-6xl gap-10 px-4 py-16 sm:px-6 lg:grid-cols-2 lg:items-start lg:gap-14 lg:py-20",
			children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "flex min-w-0 flex-col gap-5",
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "text-xs font-medium tracking-[0.22em] text-muted uppercase",
						children: "The operator"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h2", {
						className: "font-display text-4xl leading-tight font-medium tracking-[-0.03em] sm:text-5xl",
						children: "The source-line"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "text-base leading-relaxed text-muted",
						children: "The whole journey is one idea repeated: turn text into numbers, predict the next token, then keep changing the data and the loss until the model does what we want. Fareed Khan’s repo occupies that line from The Pile to chat, in plain PyTorch."
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
						className: "font-mono text-sm break-words text-fg",
						children: [
							"τ : text → tokens → Transformer → CE → base → SFT → ",
							"{",
							"PPO, DPO",
							"}",
							" → GRPO → chat"
						]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "flex flex-wrap items-center gap-3",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
							type: "button",
							onClick: () => {
								setCursor(0);
								setPlaying(true);
							},
							className: "inline-flex h-11 items-center rounded-md bg-accent px-4 text-sm font-medium text-accent-fg transition-[scale,background-color] duration-150 ease-out hover:bg-fg active:scale-[0.96]",
							children: "Walk the source-line"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
							type: "button",
							onClick: () => {
								setPlaying(false);
								setCursor(0);
							},
							className: "inline-flex h-11 items-center rounded-md px-4 text-sm font-medium text-fg shadow-[var(--shadow-border)] transition-[scale,box-shadow,background-color] duration-150 ease-out hover:bg-elevated/60 active:scale-[0.96]",
							children: "Reset"
						})]
					})
				]
			}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("aside", {
				className: "rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6",
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "font-mono text-xs tracking-[0.18em] text-subtle uppercase",
						children: "FN · eleven seats"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("ol", {
						className: "mt-5 flex flex-wrap gap-2",
						children: PIPELINE.map((step, i) => {
							const active = i === cursor;
							return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("li", {
								className: "relative",
								children: [active ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", { className: "dwell-ring" }) : null, /* @__PURE__ */ (0, import_jsx_runtime.jsx)("button", {
									type: "button",
									onClick: () => {
										setPlaying(false);
										setCursor(i);
									},
									className: cn("relative h-10 rounded-md px-3 font-mono text-sm tabular-nums shadow-[var(--shadow-border)] transition-[background-color,box-shadow,color] duration-150 ease-out", active ? "dwell-beat bg-elevated text-fg" : "text-muted hover:bg-elevated hover:text-fg", step.sink && active && "text-mark"),
									children: step.label
								})]
							}, step.id);
						})
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
						className: "mt-5 font-display text-2xl leading-snug font-medium tracking-[-0.02em] italic text-fg",
						children: [current.label, current.sink ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
							className: "text-mark",
							children: " ∞"
						}) : null]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "mt-2 text-sm leading-relaxed text-muted",
						children: current.hint
					})
				]
			})]
		}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
			className: "mx-auto max-w-6xl px-4 pb-16 sm:px-6 lg:pb-20",
			children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6",
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
						className: "flex flex-wrap items-end justify-between gap-3",
						children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
							className: "text-xs font-medium tracking-[0.22em] text-muted uppercase",
							children: "Tokenize"
						}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("h3", {
							className: "font-display mt-2 text-2xl font-medium tracking-[-0.02em]",
							children: "Write n as pieces"
						})] }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
							className: "font-mono text-xs text-subtle",
							children: [tokens.length, " pieces · stand-in IDs, not r50k_base"]
						})]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("label", {
						htmlFor: "tok-input",
						className: "mt-5 block text-sm font-medium text-fg",
						children: "Raw text"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("textarea", {
						id: "tok-input",
						value: text,
						onChange: (e) => setText(e.target.value),
						rows: 3,
						className: "mt-2 w-full resize-y rounded-md bg-elevated px-3 py-2.5 font-mono text-sm text-fg shadow-[var(--shadow-border)] outline-none placeholder:text-subtle focus-visible:ring-2 focus-visible:ring-ring/70"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("p", {
						className: "mt-2 text-xs leading-relaxed text-subtle",
						children: [
							"The repo uses tiktoken ",
							/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "text-fg",
								children: "r50k_base"
							}),
							" (50 257 tokens, padded to 50 304). This box only splits on the GPT-2 regex and hashes each piece so you can see the cut. The real BPE table lives in the training scripts."
						]
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", {
						className: "mt-4 flex flex-wrap gap-1.5",
						children: tokens.length === 0 ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
							className: "text-sm text-subtle",
							children: "Type something to see the cut."
						}) : tokens.map((tok, i) => /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("span", {
							className: "token-chip",
							children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "text-fg",
								children: tok.piece === " " ? "␣" : tok.piece
							}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
								className: "text-subtle tabular-nums",
								children: tok.id
							})]
						}, `${tok.id}-${i}`))
					})
				]
			})
		})]
	});
}
var CLONE = `git clone https://github.com/FareedKhan-dev/train-llm-from-scratch.git
cd train-llm-from-scratch
pip install -e .`;
var EXTRAS = `pip install -e ".[train]"   # datasets + wandb
pip install -e ".[ui]"      # streamlit control panel
pip install -e ".[docs]"    # mkdocs
pip install -e ".[all]"`;
var RUN = `NPROC=1 bash scripts/run_posttraining.sh`;
function CopyBlock({ label, code }) {
	const [copied, setCopied] = (0, import_react.useState)(false);
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
		className: "rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6",
		children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "flex items-center justify-between gap-3",
			children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
				className: "font-mono text-xs tracking-[0.18em] text-subtle uppercase",
				children: label
			}), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("button", {
				type: "button",
				onClick: async () => {
					try {
						await navigator.clipboard.writeText(code);
						setCopied(true);
						window.setTimeout(() => setCopied(false), 1400);
					} catch {
						setCopied(false);
					}
				},
				className: "inline-flex h-10 items-center gap-2 rounded-md px-3 text-sm text-muted shadow-[var(--shadow-border)] transition-[background-color,color] duration-150 ease-out hover:bg-elevated hover:text-fg",
				"aria-label": `Copy ${label}`,
				children: [copied ? /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Check, {
					className: "size-4",
					strokeWidth: 2
				}) : /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Copy, {
					className: "size-4",
					strokeWidth: 2
				}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("span", {
					className: "font-mono text-xs",
					children: copied ? "copied" : "copy"
				})]
			})]
		}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("pre", {
			className: "mt-4 overflow-x-auto font-mono text-xs leading-relaxed text-fg sm:text-sm",
			children: code
		})]
	});
}
function RepoSection() {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsx)("section", {
		id: "repo",
		className: "scroll-mt-16 border-b border-border",
		children: /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
			className: "mx-auto grid max-w-6xl gap-10 px-4 py-16 sm:px-6 lg:grid-cols-2 lg:items-start lg:gap-16 lg:py-20",
			children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", { children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "text-xs font-medium tracking-[0.22em] text-muted uppercase",
					children: "The sink amount"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("h2", {
					className: "font-display mt-3 text-4xl leading-tight font-medium tracking-[-0.03em] sm:text-5xl",
					children: "The repository"
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "mt-5 text-base leading-relaxed text-muted",
					children: "Twelve was the unique sink of An. Chat is the unique sink of τ. The asked gift is the same number the README already asks: clone the repo, run the smoke configs, read the files. Stars are optional. The MIT license is not."
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
					className: "mt-4 text-sm leading-relaxed text-subtle",
					children: "Fareed Khan is looking for a PhD position in AI. The work is a tutorial that grew into a full post-training line — still one idea, repeated."
				}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
					className: "mt-6 flex flex-col gap-2 sm:flex-row",
					children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
						href: REPO,
						target: "_blank",
						rel: "noopener noreferrer",
						className: "inline-flex h-12 flex-1 items-center justify-center rounded-md bg-accent px-4 text-sm font-medium text-accent-fg no-underline transition-[scale,background-color] duration-150 ease-out hover:bg-fg active:scale-[0.96]",
						children: "Open the GitHub repo"
					}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("a", {
						href: DOCS,
						target: "_blank",
						rel: "noopener noreferrer",
						className: "inline-flex h-12 flex-1 items-center justify-center rounded-md px-4 text-sm font-medium text-fg no-underline shadow-[var(--shadow-border)] transition-[scale,box-shadow,background-color] duration-150 ease-out hover:bg-elevated/60 hover:shadow-[var(--shadow-border-hover)] active:scale-[0.96]",
						children: "Read the docs"
					})]
				})
			] }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
				className: "flex flex-col gap-4",
				children: [
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "font-display text-5xl leading-none font-medium tracking-[-0.04em] tabular-nums text-fg",
						children: "9.6k"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)("p", {
						className: "text-sm text-muted",
						children: "stars · 1.3k forks · MIT"
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(CopyBlock, {
						label: "Setup",
						code: CLONE
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(CopyBlock, {
						label: "Extras",
						code: EXTRAS
					}),
					/* @__PURE__ */ (0, import_jsx_runtime.jsx)(CopyBlock, {
						label: "Run the line",
						code: RUN
					})
				]
			})]
		})
	});
}
function Home() {
	return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("div", {
		className: "min-h-dvh overflow-x-clip bg-bg text-fg",
		children: [
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)(SiteHeader, {}),
			/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("main", { children: [
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Hero, {}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(PathSection, {}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(ModelSection, {}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(LossSection, {}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(AlignSection, {}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(EvalSection, {}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(NameSection, {}),
				/* @__PURE__ */ (0, import_jsx_runtime.jsx)(RepoSection, {})
			] }),
			/* @__PURE__ */ (0, import_jsx_runtime.jsx)(SiteFooter, {})
		]
	});
}
//#endregion
export { Home as component };
