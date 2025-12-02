# Domain Adaptation for Spanish→English MT with Llama 3.1

This repository contains the artifacts for a Machine Translation project that probes how far large instruction-tuned models such as **Meta-Llama-3.1-8B-Instruct** can be pushed toward a medical/regulatory domain using increasingly heavy-weight adaptation strategies:

1. **Zero-shot** prompting (baseline).
2. **In-context learning (ICL)** with a single retrieved parallel example.
3. **Retrieval-Augmented Generation (RAG)** with the top five matches from a FAISS-backed knowledge base.
4. **Parameter-efficient fine-tuning (PEFT) via QLoRA** on ~10k trusted in-domain sentence pairs.

Model quality is benchmarked with BLEU, chrF, METEOR, and COMET, while translation throughput is tracked to expose quality/speed trade-offs between retrieval and training-heavy approaches.

---

## Repository Layout

| Path | Description |
| --- | --- |
| `Baseline_Zero_shot.ipynb` | Zero-shot pipeline: prompt construction, inference, and evaluation over the `data/test/all-filtered.*.real.test` split. |
| `MT_ICL_RAG.ipynb` | End-to-end RAG notebook: builds a FAISS retriever from training data, runs batched ICL/RAG inference, and scores outputs. |
| `Fine-tuning.ipynb` | QLoRA training workflow built on [Unsloth](https://github.com/unslothai/unsloth) for the 8B Llama base model plus evaluation of the fine-tuned adapter. |
| `Lora_Translate` | Companion notebook (JSON) that loads the saved LoRA adapter for fast batched inference + scoring. |
| `data/train` | Parallel corpora used for knowledge base construction and QLoRA training (see below). |
| `data/test` | Held-out test sentences, parallel references, and stored prediction files such as `baseline_predictions.txt`. |
| `Machine Translation - Final Report.docx` | Narrative report summarizing findings. |

---

## Data Organization

All corpora are plain-text, UTF-8, one sentence per line. Filenames encode language (`en` or `es`), quality (`real` curated alignments vs. `fuzzy` alignments), and split:

| File | Size | Usage |
| --- | --- | --- |
| `data/train/all-filtered.es.real.smalltrain` | 10,000 lines | Primary Spanish source sentences for training, retrieval, and evaluation prompts. |
| `data/train/all-filtered.en.real.smalltrain` | 10,000 lines | Matching English references. |
| `data/train/all-filtered.*.fuzzy.smalltrain` | 50,000 lines per language | Additional noisy pairs available for experimentation but **not** used in the current notebooks. |
| `data/test/all-filtered.es.real.test` | 10,000 lines (ICL/RAG use 5,000) | Held-out Spanish inputs. |
| `data/test/all-filtered.en.real.test` | 10,000 lines | Gold English references. |
| `data/test/all-filtered.*.fuzzy.test` | 50,000 lines per language | Optional out-of-domain evaluation. |
| `data/test/baseline_predictions.txt` | 5,000 lines | Stored zero-shot outputs aligned with the first 5k references. |

👉 **Conventions**

- When notebooks down-select to 5,000 sentences, they trim every list (`sources`, `references`, `predictions`) to the same length before scoring.
- Generated hypotheses for new experiments should be saved under `data/test/<custom_name>.txt` so the evaluation cells can point to them easily.

---

## Environment & Setup

1. **Hardware**: All workflows assume access to a CUDA GPU (T4 for QLoRA training, A100 for RAG experiments). CPU-only runs are not realistic for these models.
2. **Python**: 3.10+ recommended. Create an isolated environment, e.g.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Dependencies**: Install the core stack (the versions below reflect what was used in Colab; feel free to pin as needed).
   ```bash
   pip install --upgrade pip
   pip install "unsloth>=2025.11.4" "transformers>=4.57.2" datasets sentence-transformers faiss-cpu \
       sacrebleu unbabel-comet nltk pandas scikit-learn peft accelerate bitsandbytes entmax torch torchvision torchaudio
   ```
   - `unbabel-comet` pulls a heavy checkpoint the first time you score COMET.
   - `faiss-cpu` is enough unless you plan to rebuild the retriever on-GPU.
4. **NLTK assets**: The notebooks download `punkt`, `punkt_tab`, and `wordnet`. If you run locally, pre-download them via:
   ```bash
   python -m nltk.downloader punkt punkt_tab wordnet
   ```
5. **Hugging Face Authentication**: Llama weights require an access token.
   ```bash
   huggingface-cli login
   export HF_TOKEN=<token>  # if you prefer env vars inside notebooks
   ```

---

## Running the Experiments

All workflows are written as Jupyter notebooks. They can run in Google Colab (original environment) or locally if your GPU has at least 14 GB of VRAM for inference and ~16 GB for QLoRA training.

### 1. Zero-Shot Baseline (`Baseline_Zero_shot.ipynb`)

1. Authenticate with Hugging Face and mount storage if you are on Colab.
2. Update the `DATA_DIR`, `TEST_SRC`, and `TEST_REF` paths if your directory layout differs.
3. Run the model loading cell to pull `Meta-Llama-3.1-8B-Instruct` (4-bit through Unsloth) onto the GPU.
4. The `translate_batch` helper constructs chat prompts of the form “Translate this Spanish sentence…”. Adjust `BATCH_SIZE` and `MAX_NEW_TOKENS` to trade speed vs. stability.
5. Export predictions (default: `/content/drive/.../all-filtered.es.real.test.llama31.en`) and run the scoring block to compute BLEU/chrF/METEOR/COMET. Scores in the checked-in run: **BLEU 33.4 / chrF 65.1 / METEOR 0.642 / COMET 0.830**.

### 2. Retrieval & ICL/RAG (`MT_ICL_RAG.ipynb`)

1. Install `sentence-transformers`, `faiss`, `unsloth`, and evaluation libraries via the first cell (restart the runtime when prompted).
2. `TranslationRetriever` loads the 10k `real` pairs, builds sentence embeddings using `paraphrase-multilingual-MiniLM-L12-v2`, normalizes them, and writes them into a FAISS inner-product index. Building the index once saves ~8 minutes per run.
3. The notebook defines two prompt builders:
   - **ICL**: injects the single most similar (Spanish, English) example into the prompt.
   - **RAG**: passes the top five retrieved pairs inside a “Reference translations” block.
4. `translate_all_icl_batched` and `translate_all_rag_batched` run the batched inference loops (adjust `batch_size`, `max_new_tokens`, `temperature`, `top_p` as desired).
5. Evaluation uses `evaluate_all_metrics`, which wraps SacreBLEU/chrF, METEOR, and COMET. Outputs are written to disk for traceability.
6. Observed metrics on 5k sentences:  
   - **ICL (top-1)**: BLEU 47.5 / chrF 71.4 / METEOR 0.729 / COMET 0.872   
   - **RAG (top-5)**: BLEU 48.2 / chrF 72.0 / METEOR 0.736 / COMET 0.875 

### 3. QLoRA Fine-Tuning (`Fine-tuning.ipynb`)

1. Read the 10k `real` training pairs into memory, shuffle/split with `train_test_split` (90/10).
2. `FastLanguageModel.from_pretrained` loads the 4-bit base model, then `FastLanguageModel.get_peft_model` applies a LoRA adapter (rank 8, `lora_alpha=16`, dropout 0.05) on attention projections + MLPs.
3. Training data is formatted with `tokenizer.apply_chat_template` so the model learns to respond as an assistant translator.
4. `SFTTrainer` handles instruction-tuning using `gradient_accumulation_steps=8` (effective batch size 64), learning rate `2e-4`, `num_train_epochs=3`, and bf16/FP16 mixed precision depending on hardware.
5. The adapter and tokenizer are saved to `/content/drive/.../llama31_es_en_lora`, which is what the `Lora_Translate` notebook consumes for inference.
6. Sanity-check translations show clean, fluent outputs; full test-set evaluation (see below) beats the baselines at modest training cost (~21M trainable params).

### 4. LoRA Inference & Scoring (`Lora_Translate`)

This notebook mirrors the baseline scoring flow but swaps in the fine-tuned adapter. It:

1. Loads the saved LoRA weights and tokenizer.
2. Runs batched decoding on the test Spanish set.
3. Writes the outputs to `/content/drive/.../all-filtered.es.real.test.llama31_lora.en`.
4. Computes BLEU/chrF/METEOR/COMET for direct comparison. Recorded metrics: **BLEU 49.0 / chrF 72.1 / METEOR 0.750 / COMET 0.879**, with the first-line translation matching domain-specific phrasing.

---

## Evaluation Workflow

All notebooks share a consistent evaluation block:

1. **Load system translations** from disk, stripping trailing whitespace.
2. **Load references and optionally sources** from `data/test`.
3. Align lengths with `n = min(len(...))`.
4. Compute:
   ```python
   bleu = sacrebleu.corpus_bleu(sys_lines, [ref_lines])
   chrf = sacrebleu.corpus_chrf(sys_lines, [ref_lines])
   meteor_scores = [meteor_score([word_tokenize(ref)], word_tokenize(hyp)) ...]
   comet_output = comet_model.predict([...], batch_size=64, gpus=1)
   ```
5. Print + log the aggregate metrics. COMET downloads `Unbabel/wmt22-comet-da` on the first call; cache reuse makes subsequent runs faster.

You can reuse the evaluation cell standalone by pointing `OUT_PATH`, `REF_PATH`, and `SRC_PATH` to new files.

---

## Results Summary

| Approach | Data Touchpoint | BLEU ↑ | chrF ↑ | METEOR ↑ | COMET ↑ | Throughput |
| --- | --- | --- | --- | --- | --- | --- |
| Zero-shot Llama 3.1 | None beyond prompt | 33.4 | 65.1 | 0.642 | 0.830 | 2.8 sent/s |
| ICL (top-1) | FAISS retrieval over 50k pairs | 47.5 | 71.4 | 0.729 | 0.872 | 2.4 sent/s |
| RAG (top-5) | Same retriever | 48.2 | 72.0 | 0.736 | 0.875 | 2.1 sent/s |
| QLoRA (10k pairs) | Full PEFT training | **49.0** | **72.1** | **0.750** | **0.879** | 3.2 sent/s (with batching) |

Key takeaways:

- Even a single retrieved example closes ~14 BLEU vs. zero-shot.
- RAG gains a modest +0.7 BLEU with better batching throughput after amortizing prompt cost.
- QLoRA is the only method that breaks 49 BLEU / 0.75 METEOR, but it requires several hours of GPU time upfront; 

---

## Next Steps

- Extend the dataset to include the 50k `fuzzy` pairs and study how noise impacts retrieval vs. fine-tuning.
- Convert the notebooks into Python scripts or a Lightning pipeline for reproducible training and evaluation on local clusters.
- Automate metric logging (e.g., MLflow or Weights & Biases) for easier comparison as new variants are added.

---

## Git / GitHub Initialization (commands only — do **not** run yet)

```bash
# inside /Users/koushik/Documents/Fall\ 2025/Machine\ Translation/MT_Project
git init
git add .
git commit -m "Initial commit: MT domain adaptation experiments"
git branch -M main
git remote add origin git@github.com:<your-username>/<new-repo-name>.git
git push -u origin main
```

Replace `<your-username>` and `<new-repo-name>` with your GitHub handle and the repository you create in the GitHub UI.

---

Feel free to reach out or open issues once this repository lives on GitHub—suggestions for new adaptation techniques (prompt tuning, self-alignment, etc.) are very welcome. Happy translating!
