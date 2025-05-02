# ── train_albert_squad.py ───────────────────────────────────────────
"""
Fine‑tune ALBERT‑base‑v2 on SQuAD 2.0 for as many epochs as possible within 5 hours
and save the model plus JSON logs of the training‑ and validation‑set metrics.

CPU‑only is fine; if you have an Apple‑silicon GPU, PyTorch ≥ 2.1
installs Metal‑accelerated wheels automatically.

Usage:
    python train_albert_squad.py
"""

import json, os, collections, numpy as np
import time
from datasets import load_dataset
from transformers import (
    AlbertTokenizerFast,
    AlbertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
    TrainerCallback,
)
import evaluate

# ───────────────────────────────
# 1 Data
# ───────────────────────────────
raw_dsets = load_dataset("squad_v2")        # train / validation splits

# ───────────────────────────────
# 2 Tokenizer & model
# ───────────────────────────────
checkpoint = "albert-base-v2"
tokenizer  = AlbertTokenizerFast.from_pretrained(checkpoint)
model      = AlbertForQuestionAnswering.from_pretrained(checkpoint)

max_len   = 384
doc_stride = 128
pad_on_right = tokenizer.padding_side == "right"
cls_index = tokenizer.cls_token_id


def prepare_features(examples):
    # Tokenize question‑context pairs; stride handles long contexts
    tokenized = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context"  if pad_on_right else "question"],
        truncation="longest_first",
        max_length=max_len,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map   = tokenized.pop("overflow_to_sample_mapping")
    offset_map   = tokenized["offset_mapping"]

    start_pos, end_pos = [], []
    example_ids = []  # Add list to store example IDs

    for i, offsets in enumerate(offset_map):
        input_ids    = tokenized["input_ids"][i]
        seq_ids      = tokenized.sequence_ids(i)
        sample_idx   = sample_map[i]
        answers      = examples["answers"][sample_idx]
        example_ids.append(examples["id"][sample_idx])  # Store the example ID

        if len(answers["answer_start"]) == 0:            # impossible Q
            start_pos.append(cls_index)
            end_pos.append(cls_index)
            continue

        start_char   = answers["answer_start"][0]
        end_char     = start_char + len(answers["text"][0])

        # Identify token span in this feature that fully contains the answer
        token_start, token_end = 0, 0
        context_tok_idxs = [k for k, sid in enumerate(seq_ids) if sid == (1 if pad_on_right else 0)]

        if not context_tok_idxs:
            start_pos.append(cls_index)
            end_pos.append(cls_index)
            continue

        # tokens that belong to the context
        context_start = context_tok_idxs[0]
        context_end   = context_tok_idxs[-1]

        if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
            start_pos.append(cls_index)
            end_pos.append(cls_index)
        else:
            while context_start <= context_end and offsets[context_start][0] <= start_char:
                context_start += 1
            token_start = context_start - 1

            while context_end >= context_start and offsets[context_end][1] >= end_char:
                context_end -= 1
            token_end = context_end + 1

            start_pos.append(token_start)
            end_pos.append(token_end)

    tokenized["start_positions"] = start_pos
    tokenized["end_positions"]   = end_pos
    tokenized["example_id"] = example_ids  # Add example IDs to tokenized data
    # keep offset mapping for evaluation
    return tokenized


tokenized_dsets = raw_dsets.map(
    prepare_features,
    batched=True,
    remove_columns=raw_dsets["train"].column_names,
)

train_ds = tokenized_dsets["train"]
eval_ds  = tokenized_dsets["validation"]

# ───────────────────────────────
# 3 Metrics helper (EM & F1)
# ───────────────────────────────
squad_metric = evaluate.load("squad_v2")


def postprocess(predictions, features, raw_examples):
    """Convert start/end logits to answer strings (huggingface style)."""
    start_logits, end_logits = predictions

    example_id_to_index = {k: i for i, k in enumerate(raw_examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feat in enumerate(features):
        # Use the stored example_id instead of feat["id"]
        example_id = feat["example_id"] if "example_id" in feat else None
        if example_id is None:
            # Fallback: Use the index directly if example_id is not available
            example_idx = i % len(raw_examples)
            features_per_example[example_idx].append(i)
        else:
            features_per_example[example_id_to_index[example_id]].append(i)

    final_preds = collections.OrderedDict()
    n_best, max_answer_len = 20, 30

    for example_index, example in enumerate(raw_examples):
        feature_indices = features_per_example[example_index]
        context = example["context"]

        prelim = []
        for fi in feature_indices:
            start_log = start_logits[fi]
            end_log   = end_logits[fi]
            offsets   = features["offset_mapping"][fi]

            for s in np.argsort(start_log)[-n_best:]:
                for e in np.argsort(end_log)[-n_best:]:
                    if (
                        e < s
                        or e - s + 1 > max_answer_len
                        or offsets[s] is None
                        or offsets[e] is None
                    ):
                        continue
                    prelim.append(
                        {
                            "score": start_log[s] + end_log[e],
                            "start": offsets[s][0],
                            "end":   offsets[e][1],
                        }
                    )

        if prelim:
            best = max(prelim, key=lambda x: x["score"])
            pred_text = context[best["start"]: best["end"]]
        else:
            pred_text = ""  # predict no‑answer

        final_preds[example["id"]] = pred_text

    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in raw_examples]
    return final_preds, references


def compute_metrics(eval_pred):
    preds, _ = eval_pred
    preds_text, refs = postprocess(preds, eval_ds, raw_dsets["validation"])
    return squad_metric.compute(predictions=preds_text, references=refs)


# 5-hour time limit function
class TimeConstraintCallback(TrainerCallback):
    def __init__(self, time_limit_hours=5):
        self.time_limit_secs = time_limit_hours * 60 * 60
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        if elapsed > self.time_limit_secs:
            control.should_training_stop = True
            print(f"⏱️  Time limit of {self.time_limit_secs/3600:.1f} hours reached. Stopping training.")
        return control


# ───────────────────────────────
# 4 Training
# ───────────────────────────────
args = TrainingArguments(
    output_dir="albert_squad2_finetuned",
    num_train_epochs=5,         # Will stop earlier if time limit is reached
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_accumulation_steps=2,
    learning_rate=3e-5,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    eval_steps=500,
    do_eval=True,
    save_steps=500,
    save_total_limit=2,         # Only keep the 2 most recent checkpoints
    logging_strategy="steps",
    logging_steps=50,
    report_to="none",   # no wandb etc.
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[TimeConstraintCallback(time_limit_hours=5)],
)

train_out   = trainer.train()
eval_metrics = trainer.evaluate()

# ───────────────────────────────
# 5 Save artefacts & metrics
# ───────────────────────────────
trainer.save_model(args.output_dir)              # model + tokenizer
tokenizer.save_pretrained(args.output_dir)

with open(os.path.join(args.output_dir, "train_metrics.json"), "w") as f:
    json.dump(train_out.metrics, f, indent=2)
with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
    json.dump(eval_metrics, f, indent=2)

print("✅  Finished fine‑tuning. Model & logs saved to:", args.output_dir) 