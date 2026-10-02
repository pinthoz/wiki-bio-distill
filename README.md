<p align="center"><img src="frontend/logo.svg" width="96" alt=""></p>

# wiki-bio-distill

Knowledge distillation for structured extraction: a 7B teacher labels Portuguese Wikipedia biographies, and two small students learn to extract the same JSON. One is a generative Qwen 1.5B fine-tuned with QLoRA; the other is a BERT token classifier served from AWS Lambda for a few milliseconds per request.

```text
José Saramago (Azinhaga, 16 de novembro de 1922 — Tías, 18 de junho de 2010) foi um escritor … português.
```
```json
{
  "name": "José Saramago",
  "birth_date": "1922-11-16", "birth_place": "Azinhaga",
  "death_date": "2010-06-18", "death_place": "Tías",
  "nationality": ["português"],
  "occupations": ["escritor", "…"]
}
```

## How it works

1. **Collect** (`collect.py`): about 3000 biography introductions from the Portuguese Wikipedia dump on Hugging Face (`wikimedia/wikipedia`, `20231101.pt`), kept by a regex that requires a "Place, date" parenthesis and rejects events.
2. **Label with the teacher** (`label_local.py`, `notebook/01_label_teacher.ipynb`): Qwen2.5-7B-Instruct in 4 bits on a Colab T4, prompted with the schema from `common.py`. Outputs that don't validate against the schema are kept as errors and left out of training (7 of 3000, 0.23%).
3. **Gold set and split** (`review.py`, `split.py`): 150 examples for the test set, 150 for validation, 2693 for training. Test examples never enter training.
4. **Student 1, generative** (`notebook/02_train_student.ipynb`, `train_qlora.py`): Qwen2.5-1.5B-Instruct with QLoRA (4-bit NF4 base, LoRA r=16 on all linear layers), 2 epochs. It gets a one-line prompt: the rules live in the weights.
5. **Student 2, token classifier** (`bert_data.py`, `bert_train.py`): BERTimbau (`neuralmind/bert-base-portuguese-cased`). The teacher's JSON is aligned back to spans in the text (88% of training examples align fully; the rest are left out), the model tags tokens with BIO labels, and code rebuilds the JSON (dates to ISO, feminine and plural forms to the lemma).
6. **Evaluate** (`evaluate.py`): precision, recall and F1 per field against the gold set, with values compared as normalized sets. A hallucinated value counts as a false positive.
7. **Export and serve**: the BERT goes to ONNX (fp32, and int8 weight-only) and runs on Lambda behind API Gateway; the Qwen student goes to GGUF Q4_K_M for llama.cpp.

## Results

Test set: 150 biographies (dataset `v1`).

| Model | Valid JSON | F1 | Precision | Recall | F1 nationality | F1 occupations |
|---|---|---|---|---|---|---|
| Teacher, Qwen 7B | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Base Qwen 1.5B, detailed prompt, no training | 0.653 | 0.636 | 0.841 | 0.511 | 0.655 | 0.603 |
| **Student Qwen 1.5B + LoRA, fp16** | 1.000 | **0.932** | 0.938 | 0.926 | 0.855 | 0.909 |
| Student Qwen 1.5B, GGUF Q4_K_M, JSON grammar | 1.000 | 0.934 | 0.936 | 0.933 | 0.877 | 0.904 |
| **Student BERTimbau, PyTorch** | 1.000 | **0.924** | 0.922 | 0.926 | 0.818 | 0.892 |
| Student BERTimbau, ONNX fp32 | 1.000 | 0.924 | 0.922 | 0.926 | 0.818 | 0.892 |
| Student BERTimbau, ONNX int8 weights | 1.000 | 0.924 | 0.922 | 0.926 | 0.818 | 0.892 |

**Fine-tuning is what makes the small model useful.** With the teacher's full prompt and no training, the base 1.5B returns valid JSON only 65% of the time and misses half of the information (recall 0.511). After QLoRA, the same model returns valid JSON every time and reaches 0.932 F1, with a one-line prompt.

**4-bit quantization costs nothing here.** The GGUF Q4_K_M student (940 MB, against 3.1 GB in fp16) scores 0.934 F1, within noise of the fp16 model, with JSON guaranteed by a grammar built from the schema. It is slow on a laptop CPU, though: a median of 11.9 s per biography with llama.cpp, against 29 ms for the BERT on a laptop GPU and about 65 ms for the BERT on Lambda. That is why the BERT is the one serving the API.

**Read the teacher row with care.** The gold labels were accepted from the teacher without manual review (`review.py --accept-all`), so the test measures agreement with the teacher, not correctness: the teacher scores ~1.0 by construction, and its mistakes are not counted against the students. Some of the BERT "errors" are cases where BERT is right and the teacher is not (a nationality the text never states, a country left in a place name). Reviewing the gold set by hand is the main open item.

### What the errors look like

- **Nationality and occupations are the weak fields** for both students. The teacher is inconsistent with gender (`brasileira` 96 times against `brasileiro` 890, `atriz` 185 against `ator` 223), so the BERT's learned lemma map keeps some feminine forms.
- **BERT can only point at text that exists.** It never invents a value, but sometimes tags a piece of a word (`depu`**`tada`**, `supermer`**`cad`**`ista`) or cuts a name short.
- **The Qwen student writes the JSON itself.** It normalizes gender as the prompt asks, but occasionally adds or merges information (`"infante"` as an occupation, `"Santa Cruz do Capibaribe/PE"` as a place).
- BERT and Qwen disagree on 70 of the 150 test texts; BERT fp32 and BERT int8 disagree on 1.

### Quantizing the BERT

ONNX Runtime's default `quantize_dynamic` (int8 weights and int8 activations) broke the model exported with the `torch.export`-based exporter: F1 fell from 0.924 to 0.460, with the same tags as PyTorch on only 89.6% of tokens. Weight-only int8 with `MatMulNBits` (blocks of 32, activations kept in fp32) kept every tag on the 51 texts checked and the same F1, at 190 MB instead of 435 MB.

## Serving on AWS

```text
Browser ──▶ CloudFront ──"/"────────▶ S3 site bucket (private)
                  └──"/extract*"──▶ API Gateway (HTTP API, 2 req/s)
                                       ├─ Lambda authorizer: x-api-key vs key in SSM
                                       ├─▶ Lambda BERT fp32 ┐  container image from ECR,
                                       ├─▶ Lambda BERT int8 ┘  ONNX model loaded from S3
                                       └─▶ EC2 t3.small (optional, same code, secret header)
```

- **The page** (`frontend/index.html`) shows each model's result as a trading card, with the extracted spans highlighted in the text. It calls the API on its own CloudFront domain, so there is no CORS.
- **The Lambdas** run the same container image; `MODEL_FILE` picks the model. At start-up each one downloads its ONNX model from S3 straight into memory: reading the 435 MB model from the container image ran at ~6 MB/s and took over a minute; from S3 it takes 3.4 s.
- **The EC2 server** (`app_bert/server.py`, `infra/api_ec2.tf`) wraps the same Lambda handler in a plain HTTP server, managed by systemd. API Gateway adds a secret header, so calling the instance directly gets a 403. It is off by default (`-var ec2_api=true`).
- **Everything is Terraform** (`infra/`), including the API key (a random value in SSM SecureString).

### Lambda or EC2: measured

Ten requests per route from a laptop in Portugal to `eu-west-1`, through API Gateway with the API key. "Model" is the inference time reported by the server; the rest is network, API Gateway and the authorizer.

| Route | First request, cold | Following requests (median) | Model only (median) | Monthly cost with no traffic |
|---|---|---|---|---|
| Lambda, BERT fp32 (3008 MB) | 6.14 s | 0.30 s | 61 ms | ~0 |
| Lambda, BERT int8 weights (3008 MB) | 3.42 s | 0.48 s | 248 ms | ~0 |
| EC2 `t3.small`, BERT fp32 | no cold start (0.29 s) | 0.30 s | 75 ms | ~20 USD |

- Once warm, Lambda fp32 and EC2 feel the same to the user: about 230 ms of each request is the network and API Gateway.
- The EC2 server only wins on the first request. Its burstable CPU is slightly slower and less steady (outliers of 139 and 238 ms).
- The int8 model starts in half the time but runs about 4× slower per request on CPU.
- **Break-even, approximate:** at Lambda's `eu-west-1` x86 prices, one fp32 request costs about 0.0000034 USD, so the always-on server only pays for itself above roughly **6 million requests a month** (about 1.6 million for the int8 model). For this project's traffic, Lambda wins; to remove the cold start, provisioned concurrency is cheaper than a server.

## Repository

| Path | What it is |
|---|---|
| `common.py` | Schema (pydantic), prompts, JSON parsing and metrics, shared by every step |
| `collect.py`, `label_local.py`, `review.py`, `split.py` | Data: collect, label, gold set, split |
| `gen.py`, `evaluate.py`, `eval_gguf.py` | Batched generation with timing; comparison table against the gold set; the GGUF student on CPU |
| `train_qlora.py` | QLoRA training as a script, with checkpoints synced to S3 (for EC2 Spot) |
| `bert_data.py`, `bert_train.py` | JSON ↔ spans alignment; BERT training, evaluation and ONNX export |
| `notebook/` | Colab notebooks, in order: `01_label_teacher` (teacher, gold set, split), `02_train_student` (baseline, QLoRA, evaluation, GGUF), `03_bert_student` (runs `bert_train.py`) |
| `app_bert/` | Lambda handler, Dockerfile and the EC2 HTTP server |
| `frontend/` | The card page and its deploy script |
| `infra/` | Terraform: data bucket, Lambdas, API Gateway and authorizer, CloudFront site, optional EC2 server and GPU training instance |

## Running it

- **Data and labels:** `notebook/01_label_teacher.ipynb` on a Colab T4. It reads AWS credentials from Colab Secrets, or asks for them when run from VS Code.
- **Qwen student:** `notebook/02_train_student.ipynb` on a Colab T4. Every result is copied to Drive and S3 as soon as it exists, and the cells skip finished work after a disconnect.
- **BERT student:** `python bert_train.py` on any CUDA GPU (an RTX 2060 with 6 GB is enough), or `notebook/03_bert_student.ipynb` on a Colab T4. `--fresh` retrains from scratch; `--export-only` redoes the ONNX export.
- **Infrastructure:** `terraform -chdir=infra apply -var bert_image_tag=<tag>`, then `bash frontend/deploy.sh`. `terraform output -raw api_key` prints the key the page asks for.

## Data license

The biographies come from Wikipedia and are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); each example keeps the URL of its article, and any dataset or model derived from them is shared under the same terms.

**Stack:** Qwen2.5 · QLoRA (PEFT, TRL, bitsandbytes) · BERTimbau → ONNX (fp32 + int8 weights) · llama.cpp GGUF · AWS Lambda · API Gateway + API key · S3 + CloudFront · EC2 · Terraform
