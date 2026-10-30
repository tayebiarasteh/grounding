"""
experiments/reported_statistics.py
Created on October 29, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import itertools
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from experiments._common import get_cfg, read_results_csv
from experiments.study import out

TRANSFORMERS = ["modernbert_large", "flan_t5_large", "roberta_large", "electra_large", "flan_t5_base", "modernbert_base",
                "longformer_base", "biomedbert_base"]
BRACE = [t + "_w" for t in TRANSFORMERS] + ["modernbert_large_w_seed1", "modernbert_large_w_seed2", "qwen3_8b_lora_w"]
POOL = TRANSFORMERS + ["modernbert_large_seed1", "modernbert_large_seed2", "qwen3_8b_lora"]
WITHOUT_CONTRADICTED = ["modernbert_large_noD", "modernbert_large_noD_easy", "modernbert_large_noD_random", "qwen3_8b_lora_noD",
                        "modernbert_large_noD_w", "qwen3_8b_lora_noD_w"]
PUBLIC = ["MiniCheck-Flan-T5-Large", "Bespoke-MiniCheck-7B", "AlignScore-large", "HHEM-2.1-Open", "LettuceDetect", "Lynx-8B", "GREEN"]
LLMS = ["DeepSeek-V4-Flash", "Qwen3.8-Flash-Next", "Mistral-Small-4-119B", "MedGemma-27B", "Qwen3-8B"]
GENERATORS = ["MAIRA-2", "LLaVA-Rad", "MedGemma-1.5-4B", "LLaVA-Med", "Gemma-4-31B", "Qwen3-VL-32B"]
TEST_SETS = ["mimic", "chexpert", "rexgradient", "openi", "padchest_gr"]
FINDINGS = ["atelectasis", "cardiomegaly", "consolidation", "fracture", "pleural effusion", "pneumothorax", "pulmonary edema"]
UNCAPPED, COMPARISON, TFIDF = "modernbert_large_uncapped", "modernbert_large_comparison", "tfidf_mimic"
PANEL = BRACE + POOL + [UNCAPPED] + WITHOUT_CONTRADICTED + [COMPARISON, TFIDF] + PUBLIC + LLMS
CRITERION = (0.50, 0.80)
HELDOUT_FINDINGS = ["consolidation", "pleural effusion", "atelectasis", "pulmonary edema"]
KINDS = ["states", "negates", "not_mentioned"]
MIN_PATIENTS = 30
Q = 0.9


def _tables(cfg: Dict) -> Dict[str, pd.DataFrame]:
    files = {"test_sets": ("result_tables", "test_sets.csv"), "statements": ("result_tables", "statements.csv"),
             "controls": ("result_tables", "controls.csv"), "additions": ("result_tables_added", "additions.csv"),
             "statements_added": ("result_tables_added", "statements.csv"), "bounds_added": ("result_tables_added", "bounds.csv"),
             "flags": ("result_tables_added", "flags_and_written.csv"), "statistics_added": ("result_tables_added", "statistics.csv"),
             "bounds": ("attainable_auroc", "attainable_auroc.csv"), "statistics": ("result_tables", "statistics.csv")}
    t = {k: read_results_csv(out(cfg, *v)) for k, v in files.items()}
    reader = out(cfg, "reader_analysis", "reader_analysis.csv")
    t["reader"] = read_results_csv(reader) if os.path.exists(reader) else None
    return t


def _one(df: pd.DataFrame, **kw) -> pd.Series:
    q = df
    for k, v in kw.items():
        q = q[q[k].astype(float) == v] if isinstance(v, float) else q[q[k].astype(str) == str(v)]
    if len(q) != 1:
        raise ValueError(f"expected one row for {kw}, found {len(q)}")
    return q.iloc[0]


def _interval(r: pd.Series, prefix: str):
    return float(r[f"{prefix}_mean"]), float(r[f"{prefix}_ci_low"]), float(r[f"{prefix}_ci_high"])


def _statement(t: Dict, v: str, source: str, prefix: str, definition: str):
    added = ("genbalanced", "genbalanced_heldout", "genbalanced_retrieved", "genpooled_heldout", "genpooled_retrieved", "genpooled_nollavamed")
    df = t["statements_added"] if source in added else t["statements"]
    r = _one(df, verifier=v, source=source, form="template", definition=definition, sensitivity_target=Q, threshold_basis="test")
    return _interval(r, prefix)


def _cell(t: Dict, v: str, source: str, agrees: int, support: int):
    r = _one(t["additions"], table="citation_cells", verifier=v, source=source, form="template", sensitivity_target=Q,
             report_state_agrees=float(agrees), retrieved_support=float(support))
    return float(r["accepted_mean"]), int(r["n"])


def _flagged(t: Dict, v: str) -> bool:
    return bool(_one(t["flags"], table="operating_flags", verifier=v, sensitivity_target=Q)["degenerate"])


def _metric(t: Dict, v: str, ds: str, cond: str = "unchanged"):
    return _interval(_one(t["test_sets"], table="metrics", verifier=v, dataset=ds, condition=cond, finding="all"), "auroc")


def _pairs(t: Dict, v: str, ds: str, cond: str = "unchanged", kind: str = "patient", finding: str = "all"):
    r = _one(t["test_sets"], table="pairs", verifier=v, dataset=ds, condition=cond, finding=finding)
    return _interval(r, f"{kind}_pair_acc")


def _operating(t: Dict, v: str, ds: str, prefix: str, cond: str = "unchanged"):
    r = _one(t["test_sets"], table="operating", verifier=v, dataset=ds, condition=cond, finding="all", sensitivity_target=Q)
    return _interval(r, prefix)


def _bound(t: Dict, ds: str) -> float:
    return float(_one(t["bounds"], set=f"{ds}__unchanged", channel="evidence_claim", finding="all")["auroc_bound"])


def _overlap(a, b) -> bool:
    return not (a[2] < b[1] or b[2] < a[1])


def group_means(t: Dict) -> List[Dict]:
    rows = []
    contradicted = {v: _statement(t, v, "rexerr", "contradicted_accept", "agree")[0] for v in BRACE + WITHOUT_CONTRADICTED}
    cell = {v: _cell(t, v, "rexerr_retrieved", 1, 0)[0] for v in BRACE + LLMS}
    for name, group, values, what in (
            ("contradicted ReXErr statements, training citation", BRACE, contradicted, "verifiers trained with BRACE"),
            ("contradicted ReXErr statements, training citation", WITHOUT_CONTRADICTED, contradicted, "verifiers trained without contradicted examples"),
            ("ReXErr statements with an agreeing report and an unsupporting retrieved citation", BRACE, cell, "verifiers trained with BRACE"),
            ("ReXErr statements with an agreeing report and an unsupporting retrieved citation", LLMS, cell, "zero-shot LLMs")):
        kept = [v for v in group if not _flagged(t, v)]
        vals = np.array([values[v] for v in kept])
        rows.append({"statistic": "group mean of the accepted share", "statements": name, "group": what, "verifiers": len(kept),
                     "mean": float(vals.mean()), "minimum": float(vals.min()), "maximum": float(vals.max())})
    return rows


def generator_cells(t: Dict) -> List[Dict]:
    rows = []
    verifiers = BRACE + WITHOUT_CONTRADICTED + POOL + PUBLIC + LLMS
    for v in verifiers:
        for agrees, support in itertools.product((1, 0), (1, 0)):
            vals = [c for c, n in (_cell(t, v, f"gen__{g}_retrieved", agrees, support) for g in GENERATORS) if n > 0]
            rows.append({"statistic": "generator mean of the accepted share", "verifier": v, "report_agrees": agrees,
                         "citation_supports": support, "generators": len(vals), "mean": float(np.mean(vals)) if vals else np.nan})
    return rows


def interval_comparisons(t: Dict) -> List[Dict]:
    rows = []

    def add(name, value, units, detail=""):
        rows.append({"statistic": name, "value": value, "units": units, "detail": detail})

    side = {}
    for v, ds in itertools.product(WITHOUT_CONTRADICTED, TEST_SETS):
        m, b = _metric(t, v, ds), _bound(t, ds)
        side[(v, ds)] = "below" if m[2] < b else ("above" if m[1] > b else "contains")
    for k in ("below", "contains", "above"):
        add(f"AUROC interval {k} the AUROC attainable without the report", sum(x == k for x in side.values()), len(side),
            "; ".join(f"{v} on {ds}" for (v, ds), x in side.items() if x == k and k == "above"))
    lower = [(v, ds) for v, ds in itertools.product(WITHOUT_CONTRADICTED, TEST_SETS)
             if _pairs(t, v, ds)[0] < min(_pairs(t, b, ds)[0] for b in BRACE)]
    add("patient pair accuracy below every verifier trained with BRACE", len(lower), len(WITHOUT_CONTRADICTED) * len(TEST_SETS))
    ov = [(v, b, ds) for v, b, ds in itertools.product(WITHOUT_CONTRADICTED, BRACE, TEST_SETS) if _overlap(_pairs(t, v, ds), _pairs(t, b, ds))]
    add("patient pair intervals that overlap between a verifier without and a verifier with contradicted examples", len(ov),
        len(WITHOUT_CONTRADICTED) * len(BRACE) * len(TEST_SETS), "; ".join(f"{v} and {b} on {ds}" for v, b, ds in ov))
    reach = [v for v in PUBLIC if any(_pairs(t, v, "mimic")[2] >= _pairs(t, b, "mimic")[1] for b in BRACE)]
    add("public verifiers whose patient pair interval reaches a verifier trained with BRACE on MIMIC-CXR", len(reach), len(PUBLIC), "; ".join(reach))
    ev = [(v, b) for v, b in itertools.product(PUBLIC + LLMS, BRACE)
          if _overlap(_pairs(t, v, "mimic", kind="evidence"), _pairs(t, b, "mimic", kind="evidence"))]
    add("evidence pair intervals that overlap between a public verifier or an LLM and a verifier trained with BRACE on MIMIC-CXR",
        len(ev), len(PUBLIC + LLMS) * len(BRACE))
    held = [(b, v) for b, v in itertools.product(BRACE, LLMS)
            if _overlap(_statement(t, b, "rexerr_heldout", "consistent_accept", "agree"), _statement(t, v, "rexerr_heldout", "consistent_accept", "agree"))]
    add("consistent ReXErr statements with the held-out citation, overlapping intervals of a verifier trained with BRACE and an LLM",
        len(held), len(BRACE) * len(LLMS), "; ".join(f"{b} and {v}" for b, v in held))
    pairs = list(zip(POOL, BRACE))
    up = sum(_pairs(t, b, "mimic", "heldout", finding=f)[0] > _pairs(t, a, "mimic", "heldout", finding=f)[0]
             for a, b in pairs for f in HELDOUT_FINDINGS)
    add("patient pair accuracy per finding with the held-out sentences higher after training with BRACE than on the pool", up,
        len(pairs) * len(HELDOUT_FINDINGS))
    sub = t["test_sets"][t["test_sets"]["table"].astype(str) == "subgroup"]
    for ds in ("mimic", "chexpert"):
        overlapping = 0
        for v in BRACE:
            g = sub[(sub["verifier"].astype(str) == v) & (sub["dataset"].astype(str) == ds) & (sub["n_patients"] >= MIN_PATIENTS)]
            apart = [att for att, gg in g.groupby("attribute") for (_, a), (_, b) in itertools.combinations(gg.iterrows(), 2)
                     if a["auroc_ci_high"] < b["auroc_ci_low"] or b["auroc_ci_high"] < a["auroc_ci_low"]]
            overlapping += not apart
        add(f"verifiers trained with BRACE whose subgroup AUROC intervals overlap within every attribute on {ds}", overlapping, len(BRACE))
    age = lambda v, grp: _one(sub, verifier=v, dataset="padchest_gr", attribute="age_group", group=grp)
    apart = [v for v in BRACE if age(v, "80_and_over")["auroc_ci_low"] > age(v, "40_to_59")["auroc_ci_high"]]
    add("verifiers trained with BRACE with a higher AUROC at ages 80 and over than at 40 to 59 on PadChest-GR, nonoverlapping intervals",
        len(apart), len(BRACE))
    sims = []
    for ds in TEST_SETS:
        src = t["additions"]
        lo = _interval(_one(src, table="similarity_subsets", verifier="modernbert_large_w", dataset=ds, ranking="1gram", subset="least_similar_10pct"), "auroc")
        hi = _interval(_one(src, table="similarity_subsets", verifier="modernbert_large_w", dataset=ds, ranking="1gram", subset="most_similar_10pct"), "auroc")
        if not _overlap(lo, hi):
            sims.append(ds)
    add("test sets on which the AUROC intervals of ModernBERT-large trained with BRACE on the 10% least and most similar findings do not overlap",
        len(sims), len(TEST_SETS), "; ".join(sims))
    kinds = t["test_sets"][(t["test_sets"]["table"].astype(str) == "contradiction") & (t["test_sets"]["dataset"].astype(str) == "mimic")]
    ordered, separate = 0, 0
    for v in BRACE:
        k = [_interval(_one(kinds, verifier=v, kind=x), "contradicted_accept") for x in KINDS]
        ordered += k[0][0] < k[1][0] < k[2][0]
        separate += k[0][2] < k[1][1] and k[1][2] < k[2][1]
    add("verifiers trained with BRACE whose mean accepted share is ordered states, negates, not mentioned", ordered, len(BRACE))
    add("verifiers trained with BRACE whose intervals of the three kinds do not overlap", separate, len(BRACE))
    e5 = t["statistics_added"][t["statistics_added"]["family"].astype(str) == "E5"]
    keys = ["source", "form", "definition", "sensitivity_target", "threshold_basis"]
    jj = pd.DataFrame({"pairs": e5[e5["x"].astype(str) == "patient_pairs"].set_index(keys)["value"],
                       "auroc": e5[e5["x"].astype(str) == "auroc"].set_index(keys)["value"]}).reset_index()
    training = jj[~jj["source"].astype(str).str.contains("heldout|retrieved")]
    add("variants with the training citation in which |rho| is larger with the patient pair accuracy than with the AUROC",
        int((training["pairs"].abs() > training["auroc"].abs()).sum()), len(training))
    return rows


def level_differences(t: Dict) -> List[Dict]:
    rows = []
    for a, b in zip(POOL, BRACE):
        for ds in TEST_SETS:
            rows.append({"pool": a, "brace": b, "dataset": ds, "auroc_difference": _metric(t, b, ds)[0] - _metric(t, a, ds)[0],
                         "patient_pair_difference": _pairs(t, b, ds)[0] - _pairs(t, a, ds)[0]})
    return rows


def ranks(t: Dict) -> List[Dict]:
    auroc = pd.Series({v: round(_metric(t, v, "mimic")[0], 3) for v in PANEL})
    pairs = pd.Series({v: round(_pairs(t, v, "mimic")[0], 3) for v in PANEL})
    ra, rp = auroc.rank(ascending=False, method="min"), pairs.rank(ascending=False, method="min")
    return [{"verifier": v, "auroc": auroc[v], "auroc_rank": int(ra[v]), "patient_pair_acc": pairs[v], "patient_pair_rank": int(rp[v])}
            for v in PANEL]


def outcomes(t: Dict) -> List[Dict]:
    rows = []

    def add(name, value, units, detail=""):
        rows.append({"statistic": name, "value": value, "units": units, "detail": detail})

    for cond in ("heldout_written", "heldout"):
        met = [v for v in BRACE if _operating(t, v, "mimic", "supported_accept", cond)[0] >= CRITERION[0]
               and _pairs(t, v, "mimic", cond)[0] >= CRITERION[1]]
        add(f"verifiers trained with BRACE that meet the criterion for unseen evidence, condition {cond}", len(met), len(BRACE), "; ".join(met))
    removed = {v: float(_one(t["additions"], table="class_means", verifier=v, dataset="mimic", condition="removed",
                             row_type="C")["accepted_share_tau90_point"]) for v in BRACE}
    accept = [v for v in BRACE if removed[v] > 0.5]
    add("verifiers trained with BRACE that accept most supported examples without evidence", len(accept), len(BRACE), "; ".join(accept))
    r = pd.DataFrame(ranks(t)).set_index("verifier")
    levels = BRACE + POOL + [UNCAPPED]
    seven = WITHOUT_CONTRADICTED + [TFIDF]
    add("verifiers trained on the three levels of training evidence at the top AUROC ranks", int((r.loc[levels, "auroc_rank"] <= len(levels)).sum()),
        len(levels))
    below = r.loc[seven, "auroc_rank"].between(len(levels) + 1, len(levels) + len(seven))
    add("TF-IDF baseline and verifiers trained without contradicted examples at the AUROC ranks directly below the three levels",
        int(below.sum()), len(seven))
    add("TF-IDF baseline and verifiers trained without contradicted examples at the last patient pair ranks",
        int((r.loc[seven, "patient_pair_rank"] > len(PANEL) - len(seven)).sum()), len(seven))
    st = t["statistics"]
    for src in ("radevalx", "rexval"):
        sig = [v for v in BRACE if float(_one(st, family="E6", test="kendall_tau_b", verifier=v, source=src)["p_fdr"]) < 0.05]
        add(f"verifiers trained with BRACE whose report error score is correlated with the radiologists' error counts on {src}, p_FDR < 0.05",
            len(sig), len(BRACE), "; ".join(sig))
    sig = [ds for ds in TEST_SETS if float(_one(t["test_sets"], table="paired", dataset=ds, a="qwen3_8b_lora_w", b="Qwen3-8B",
                                                 metric="patient_pair_acc")["p_fdr"]) < 0.05]
    add("test sets on which Qwen3-8B finetuned with BRACE has a higher patient pair accuracy than zero-shot, p_FDR < 0.05", len(sig),
        len(TEST_SETS), "; ".join(sig))
    rd = t["reader"]
    if rd is not None:
        pa = rd[(rd["table"].astype(str) == "reader_pairs") & (rd["packet"].astype(str) == "A_support")]
        diff = pa[pa["answer"].astype(str) != pa["answer_2"].astype(str)]
        one_state = diff[(diff["answer"].astype(str) == "neither") | (diff["answer_2"].astype(str) == "neither")]
        add("combinations of packet A answered differently by the two radiologists in which only one assigned a state",
            int(one_state["n"].sum()), int(diff["n"].sum()))
        kept = rd[(rd["table"].astype(str) == "confirmed_counts") & (rd["reader"].astype(str) == "both") & (rd["row_type"].astype(str) == "C")
                  & rd["set"].astype(str).str.endswith("__unchanged") & (rd["n_kept"] > 0)]
        states = sorted(set(zip(kept["finding"], kept["state"])))
        add("finding states with support sentences confirmed by both radiologists", len(states), 2 * len(FINDINGS),
            "; ".join(f"{f} {s}" for f, s in states))
        conf = lambda name, **kw: _one(rd[rd["table"].astype(str) == f"confirmed_{name}"], reader="both", **kw)
        met = [v for v in BRACE
               if float(conf("operating", verifier=v, dataset="mimic", condition="heldout", sensitivity_target=Q)["supported_accept_mean"]) >= CRITERION[0]
               and float(conf("pairs", verifier=v, dataset="mimic", condition="heldout", finding="all")["patient_pair_acc_mean"]) >= CRITERION[1]]
        add("verifiers trained with BRACE that meet the criterion for unseen evidence on the held-out sentences confirmed by both radiologists",
            len(met), len(BRACE), "missing: " + "; ".join(v for v in BRACE if v not in met))
    return rows


def main_reported_statistics(cfg_path: str) -> str:
    cfg = get_cfg(cfg_path)
    t = _tables(cfg)
    out_dir = out(cfg, "reported_statistics")
    os.makedirs(out_dir, exist_ok=True)
    for name, frame in (("group_means.csv", pd.DataFrame(group_means(t))), ("generator_cells.csv", pd.DataFrame(generator_cells(t))),
                        ("interval_comparisons.csv", pd.DataFrame(interval_comparisons(t))), ("outcomes.csv", pd.DataFrame(outcomes(t))),
                        ("level_differences.csv", pd.DataFrame(level_differences(t))), ("ranks.csv", pd.DataFrame(ranks(t)))):
        p = os.path.join(out_dir, name)
        tmp = f"{p}.{os.getpid()}.tmp"
        frame.to_csv(tmp, index=False)
        os.replace(tmp, p)
    return out_dir
