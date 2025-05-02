# ── train_distilbert_squad.py ─────────────────────────────────────────
"""
Fine‑tune **DistilBERT‑base‑uncased** on SQuAD 2.0 for as many epochs as
fit into a 5‑hour wall‑time budget.  
Saves the model plus JSON logs of the training‑ and validation‑set metrics.

CPU‑only is fine. On Apple‑silicon Macs, PyTorch ≥ 2.1 automatically
installs Metal‑accelerated wheels that give a nice speed‑up.

Usage:
    python train_distilbert_squad.py
"""
import json, os, collections, time
import numpy as np
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
    TrainerCallback,
)
import evaluate

# ───────────────────────────────
# 1  Data
# ───────────────────────────────
raw_dsets = load_dataset("squad_v2")          # train / validation splits

# ───────────────────────────────
# 2  Tokenizer & model
# ───────────────────────────────
checkpoint = "distilbert-base-uncased"
tokenizer  = DistilBertTokenizerFast.from_pretrained(checkpoint)
model      = DistilBertForQuestionAnswering.from_pretrained(checkpoint)

max_len    = 384
doc_stride = 128
pad_on_right = tokenizer.padding_side == "right"
cls_index  = tokenizer.cls_token_id


def prepare_features(examples):
    """Tokenise question/context pairs and create start/end position labels."""
    tokenised = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context"  if pad_on_right else "question"],
        truncation="longest_first",
        max_length=max_len,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = tokenised.pop("overflow_to_sample_mapping")
    offset_map = tokenised["offset_mapping"]

    start_pos, end_pos, example_ids = [], [], []

    for i, offsets in enumerate(offset_map):
        seq_ids    = tokenised.sequence_ids(i)
        sample_idx = sample_map[i]
        answers    = examples["answers"][sample_idx]
        example_ids.append(examples["id"][sample_idx])

        # Handle impossible questions
        if len(answers["answer_start"]) == 0:
            start_pos.append(cls_index)
            end_pos.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char   = start_char + len(answers["text"][0])

        # Context token span in this feature
        context_tok_idxs = [
            k for k, sid in enumerate(seq_ids)
            if sid == (1 if pad_on_right else 0)
        ]
        if not context_tok_idxs:
            start_pos.append(cls_index)
            end_pos.append(cls_index)
            continue

        context_start, context_end = context_tok_idxs[0], context_tok_idxs[-1]

        # If answer not fully inside this feature
        if not (offsets[context_start][0] <= start_char and
                offsets[context_end][1] >= end_char):
            start_pos.append(cls_index)
            end_pos.append(cls_index)
            continue

        # Move pointers to token boundaries
        tok_start, tok_end = context_start, context_end
        while tok_start <= tok_end and offsets[tok_start][0] <= start_char:
            tok_start += 1
        token_start = tok_start - 1

        while tok_end >= tok_start and offsets[tok_end][1] >= end_char:
            tok_end -= 1
        token_end = tok_end + 1

        start_pos.append(token_start)
        end_pos.append(token_end)

    tokenised["start_positions"] = start_pos
    tokenised["end_positions"]   = end_pos
    tokenised["example_id"]      = example_ids
    return tokenised


tokenised_dsets = raw_dsets.map(
    prepare_features,
    batched=True,
    remove_columns=raw_dsets["train"].column_names,
)

train_ds = tokenised_dsets["train"]
eval_ds  = tokenised_dsets["validation"]

# ───────────────────────────────
# 3  Metrics helper (EM & F1)
# ───────────────────────────────
squad_metric = evaluate.load("squad_v2")


def postprocess(predictions, features, raw_examples):
    """Convert start/end logits to answer strings (HF‑style)."""
    start_logits, end_logits = predictions

    # Map example‑id → feature indices
    example_id_to_index = {k: i for i, k in enumerate(raw_examples["id"])}
    feats_per_example = collections.defaultdict(list)
    for i, feat in enumerate(features):
        example_id = feat["example_id"]
        feats_per_example[example_id_to_index[example_id]].append(i)

    final_preds = collections.OrderedDict()
    n_best, max_ans_len = 20, 30

    for ex_idx, example in enumerate(raw_examples):
        prelim = []
        for fi in feats_per_example[ex_idx]:
            starts = start_logits[fi]
            ends   = end_logits[fi]
            offsets = features["offset_mapping"][fi]

            for s in np.argsort(starts)[-n_best:]:
                for e in np.argsort(ends)[-n_best:]:
                    if (e < s or e - s + 1 > max_ans_len or
                            offsets[s] is None or offsets[e] is None):
                        continue
                    prelim.append({
                        "score": starts[s] + ends[e],
                        "start": offsets[s][0],
                        "end":   offsets[e][1],
                    })

        if prelim:
            best = max(prelim, key=lambda x: x["score"])
            pred_text = example["context"][best["start"]:best["end"]]
        else:
            pred_text = ""

        final_preds[example["id"]] = pred_text

    refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in raw_examples]
    return final_preds, refs


def compute_metrics(eval_pred):
    preds, _ = eval_pred
    preds_text, refs = postprocess(preds, eval_ds, raw_dsets["validation"])
    return squad_metric.compute(predictions=preds_text, references=refs)


# ───────────────────────────────
# 4  5‑hour time‑limit callback
# ───────────────────────────────
class TimeConstraintCallback(TrainerCallback):
    def __init__(self, hours=5):
        self.limit = hours * 3600
        self.start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - self.start > self.limit:
            control.should_training_stop = True
            print(f"⏱️  {self.limit/3600:.0f}‑hour budget reached — stopping.")
        return control


# ───────────────────────────────
# 5  Training
# ───────────────────────────────
args = TrainingArguments(
    output_dir="distilbert_squad2_finetuned",
    num_train_epochs=5,                     # stopper will cut earlier if needed
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_accumulation_steps=2,
    learning_rate=3e-5,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    eval_steps=500,
    do_eval=True,
    save_steps=500,
    save_total_limit=2,                    # keep two latest checkpoints
    logging_strategy="steps",
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[TimeConstraintCallback(hours=5)],
)

train_out    = trainer.train()
eval_metrics = trainer.evaluate()

# ───────────────────────────────
# 6  Save artefacts & metrics
# ───────────────────────────────
trainer.save_model(args.output_dir)      # model + training args
tokenizer.save_pretrained(args.output_dir)

with open(os.path.join(args.output_dir, "train_metrics.json"), "w") as f:
    json.dump(train_out.metrics, f, indent=2)
with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
    json.dump(eval_metrics, f, indent=2)

print("✅  Finished fine‑tuning. Model & logs saved to:", args.output_dir)