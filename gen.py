import json, time
import torch
from common import messages


@torch.no_grad()
def generate_preds(
    model, tokenizer, rows, system, path, batch_size=8, max_new_tokens=300
):
    """Generates predictions for the test set, saving {id, raw, ms} per line."""
    tokenizer.padding_side = "left"  # batch generation: left padding
    model.eval()
    with open(path, "w", encoding="utf-8") as f:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    messages(r["text"], system),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for r in batch
            ]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            t0 = time.perf_counter()
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            ms = (
                (time.perf_counter() - t0) * 1000 / len(batch)
            )  # average time per example in the batch
            texts = tokenizer.batch_decode(
                out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for r, t in zip(batch, texts):
                f.write(
                    json.dumps(
                        {"id": r["id"], "raw": t, "ms": round(ms)}, ensure_ascii=False
                    )
                    + "\n"
                )
            print(f"{min(i + batch_size, len(rows))}/{len(rows)}")
