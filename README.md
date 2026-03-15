# Principle-Guided LLM Alignment

This repository contains the dataset, model inference scripts, and analysis code accompanying our paper on **principle-guided LLM alignment across languages**.

## Repository Structure

```
principle-llm-alignment/
├── dataset/                        # Multilingual alignment dataset (11 languages)
│   ├── llm_alignment_dataset - English.csv
│   ├── llm_alignment_dataset - Hindi.csv
│   ├── llm_alignment_dataset - Arabic.csv
│   ├── llm_alignment_dataset - Bengali.csv
│   ├── llm_alignment_dataset - Chinese.csv
│   ├── llm_alignment_dataset - French.csv
│   ├── llm_alignment_dataset - German.csv
│   ├── llm_alignment_dataset - Odia.csv
│   ├── llm_alignment_dataset - Punjabi.csv
│   ├── llm_alignment_dataset - Spanish.csv
│   └── llm_alignment_dataset - Telugu.csv
│
├── inference/                      # Model inference scripts
│   ├── hindi_inference.py          # LLM inference runner (v1)
│   ├── hindi_inference_v2.py       # LLM inference runner (v2)
│   ├── hindi_inference_v3.py       # LLM inference runner (v3, with extended models)
│   ├── phi_inference.py            # Phi-3 model inference
│   ├── gemma_27_inference.py       # Gemma 2 27B inference
│   └── run_all_alignment.sh        # Shell script to run all inference jobs
│
└── analysis/
    └── alignment_overlap_analysis.py  # Computes alignment accuracy, validity, failsafe rate, and B-bias per model and pass type
```

## Dataset

The dataset contains principle-guided alignment prompts for **11 languages**: English, Hindi, Arabic, Bengali, Chinese, French, German, Odia, Punjabi, Spanish, and Telugu.

Each row in the dataset includes:
- A **scenario** presented to an LLM with two responses (A and B)
- A **principle** guiding which response is preferable
- A **ground truth** label (A or B) indicating the correct, principle-aligned response
- LLM-generated responses and choices under four evaluation passes: Base, Base+Thinking, Principled, and Principled+Thinking

## Inference Scripts

The inference scripts run multiple LLMs (open-source via Ollama and GPT-based) on the alignment dataset and record their choices under different prompting strategies:
- **Base**: Standard prompting without any principle
- **Base+Thinking**: Base prompt plus chain-of-thought reasoning
- **Principled**: Prompt that includes the guiding principle
- **Principled+Thinking**: Principle prompt plus chain-of-thought reasoning

### Requirements

```bash
pip install ollama pandas numpy
```

To run Llama, Phi, Gemma, Qwen, Mistral, or other open-source models locally, you will need [Ollama](https://ollama.ai/) installed:

```bash
# Pull and run any model, e.g.
ollama pull llama3.1:8b
```

### Running Inference

```bash
# Run all models (requires Ollama running locally)
bash inference/run_all_alignment.sh

# Or run individual scripts:
python3 inference/hindi_inference.py
python3 inference/phi_inference.py
python3 inference/gemma_27_inference.py
```

## Analysis

The `alignment_overlap_analysis.py` script processes the output CSV from inference and computes:
- **Accuracy**: How often the model's choice matches the ground truth
- **Validity**: How often the model outputs a parseable A/B choice
- **Failsafe Rate**: Fraction of responses where retry/failsafe logic was triggered
- **B-Bias**: Fraction of responses where the model always chose B

```bash
python3 analysis/alignment_overlap_analysis.py
```

## Citation

If you use this dataset or code in your research, please cite our paper:

```bibtex
@article{anonymized,
  title={Anonymized for Review},
  year={2025}
}
```

## License

This dataset and code are released for research purposes only.
