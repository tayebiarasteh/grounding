# Automated grounding verification of AI statements about chest radiographs

This is the official repository of the paper **Automated grounding verification of AI statements about chest radiographs**.

## Overview

A grounding verifier decides whether a statement about a patient's chest radiograph is supported by the patient's radiology report and by the cited reference text. Building Report And Citation Examples (BRACE) builds training examples for such verifiers from public radiology reports with automatic labels and from reference texts. For each report finding, it builds a supported example, a contradicted example whose statement disagrees with the report but has supporting evidence, and two examples without supporting evidence. A patient pair consists of a supported example and a contradicted example that share the statement and cited evidence and differ only in the report. CheXbert labels the reports, two large language model judges select the support sentences of the Radiopaedia articles, and a third large language model writes further support sentences that the same judges accept or reject. The code trains transformer verifiers and Qwen3-8B with low-rank adaptation on these examples, scores public verifiers and zero-shot large language models on the same test examples, and evaluates every verifier with the patient pair accuracy, the evidence pair accuracy, the share of examples accepted at a fixed threshold, the evidence tests, and statements written by report generators and from ReXErr. The same construction is applied to food allergens in recipes.

## Installation

```bash
git clone https://github.com/tayebiarasteh/grounding.git
cd grounding
```

Install PyTorch, Transformers, PEFT, huggingface_hub, scikit-learn, SciPy, NumPy, pandas, pyarrow, Pillow, PyYAML, tqdm, httpx, requests, packaging, and the OpenAI Python client. LLaVA-Rad needs a clone of its repository at the path that the configuration names. MAIRA-2, LLaVA-Rad, and Bespoke-MiniCheck-7B run in a worker process with its own package folder, which the code builds with pip.

## Analyses and modules

Each entry point takes the path of `config/config.yaml`.

- **Datasets and report labels.** `main_preprocess_datasets` in `experiments/dataset_inputs.py` reads the report lists of ReXVal and RadEvalX. `main_chexbert_labels` in `experiments/report_labels.py` labels the impression of every report with CheXbert and maps its classes to the states present and absent.
- **Support sentences.** `main_judges` in `experiments/judges.py` asks each judge whether a Radiopaedia sentence concerns a finding and which state it supports. `main_support_pools` in `experiments/support_pools.py` forms the support pools and the unsupporting sentences of the training and the held-out articles. `main_write_sentences` and `main_judge_written` in `experiments/written_sentences.py` write further support and unsupporting sentences and judge them.
- **Training and test examples.** `main_build_examples` in `experiments/build_examples.py` builds the examples of every dataset and evidence condition, including the replaced and removed evidence and the replaced report. `main_build_evidence_sets` in `experiments/evidence_sets.py` builds the training sets of BRACE, the test set with the written held-out sentences, and the statements with held-out and retrieved citations.
- **AI-written statements.** `main_generate_reports` in `experiments/generate_reports.py` writes findings sections for the Open-I radiographs with each report generator. `main_statements` in `experiments/statements.py` splits the ReXErr sentences and the generated reports into statements and labels them.
- **Verifiers.** `main_train` and `main_train_added` in `experiments/train_verifiers.py` train one verifier, `main_tfidf` in `experiments/tfidf_baseline.py` fits the TF-IDF baseline, and `main_score_trained` in `experiments/score.py` and `main_score_added` in `experiments/score_added.py` score the trained verifiers. `main_public_verifiers` in `experiments/public_verifiers.py` scores the public verifiers, and `main_llm_verifiers` in `experiments/llm_verifiers.py` scores the zero-shot large language models.
- **Controls.** `main_attainable_auroc` in `experiments/attainable_auroc.py` computes the highest AUROC attainable from the evidence and the statement alone. `main_retrieval`, `main_score_retrieval`, and `main_similarity` in `experiments/controls.py` run the retrieval control and the train-test similarity.
- **Metrics and statistics.** `main_result_tables` in `experiments/result_tables.py` and `main_result_tables_added` in `experiments/result_tables_added.py` compute every metric, pair accuracy, accepted share, confidence interval, correlation, paired test, and false discovery rate correction. The correlations across the verifiers that the paper reports are those of `main_result_tables_added`, which computes them over every statement set and corrects them as one family. The correlations across the verifiers are Spearman correlations of the reported metrics. Their p-values are analytic, with the verifier as the unit. `Inference/rank_inference.py` computes the Kendall tau-b of the report error score of each verifier with the errors counted by the radiologists. Its p-value comes from a bootstrap that resamples patients and counts an undefined replicate as an exceedance. `main_reported_statistics` in `experiments/reported_statistics.py` computes the means over groups of verifiers, the shares averaged over the report generators, the comparisons stated by the overlap of confidence intervals, the counts of verifiers that meet each reported condition, the differences between the two levels of training evidence, and the ranks of the verifiers. `main_dataset_statistics` in `experiments/dataset_statistics.py` counts the datasets, the support pools, the statements, and the parameters of the models.
- **Reader study.** `main_reader_analysis` in `experiments/reader_analysis.py` analyzes the answer sheets returned by the radiologists and repeats the evaluation on the support that they confirmed. Where a patient, a report, or an evidence sentence repeats, the bootstrap of an agreement resamples that unit. The p-value of Cohen's kappa comes from this bootstrap where a unit repeats and is analytic where every item is its own unit.

Every entry point writes its results atomically, resumes an interrupted run, and skips the units it has finished. `config/serde.py` reads the configuration, `data_loader/verifier_dataset.py` builds the input text of a verifier, `controlled/train_verifier.py` trains a transformer verifier, `experiments/llm_local.py` finetunes and scores Qwen3-8B with low-rank adaptation, `Inference/llm_verifier.py` scores a large language model on a served endpoint, `Inference/chexbert_labeler.py` runs CheXbert, and `Inference/stats_utils.py` and `Inference/cluster_metrics.py` hold the bootstrap, the metrics, and the false discovery rate correction.

## Configuration and data

Replace every `/PATH` in `config/config.yaml` with the locations of the data and of the output folder. Set `hf_token` to a Hugging Face access token, and set `api_base_url` and `api_key` to an OpenAI-compatible endpoint that serves the judges, the writer, and the large language model verifiers, or set the environment variables that the configuration names. Several checkpoints require accepting their terms on Hugging Face.

For master lists generated by `main_preprocess_datasets`, set `Grounding.sources.rexval_masterlist` to `${outputs_root}/datasets_staging/XRay/ReXVal/rexval_masterlist.csv` and `Grounding.sources.radevalx_masterlist` to `${outputs_root}/datasets_staging/XRay/RadEvalX/radevalx_masterlist.csv`. These are the default staging paths written by `experiments/dataset_inputs.py`.

MIMIC-CXR, MIMIC-IV, ReXErr-v1, ReXVal, and RadEvalX are available from PhysioNet. CheXpert Plus is available from the Stanford AIMI portal, ReXGradient-160K from the Hugging Face Hub, Open-I from the US National Library of Medicine, PadChest-GR on request from BIMCV, the Radiopaedia articles on request from Radiopaedia, RecipeNLG from its project page, and the Wikipedia articles from Wikipedia. No image, report, or derived record is redistributed here.

## Citation

If you use this repository, please cite our paper:

```bibtex
@article{arasteh2026grounding,
  title  = {Automated grounding verification of AI statements about chest radiographs},
  author = {Tayebi Arasteh, Soroosh and Joodaki, Mehdi and Lotfinia, Mahshad and Nguyen, Tri-Thien and Ziegelmayer, Sebastian and Nebelung, Sven and Truhn, Daniel},
  year   = {2026}
}
```

## License

MIT License. See `LICENSE` for details.
