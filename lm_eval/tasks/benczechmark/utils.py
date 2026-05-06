from typing import Iterable, Optional

import evaluate
import numpy as np
from sklearn.metrics import f1_score

from lm_eval.api.metrics import mean
from lm_eval.api.task import ConfigurableTask, eval_logger


def aggregate_macro_f1_score(items, **kwargs):
    golds, preds = zip(*items)
    return f1_score(golds, preds, average="macro")


def avg_mcauroc(prediction, reference):
    return prediction, reference


def aggregate_avg_mcauroc(items):
    golds, probs = zip(*items)
    metric = evaluate.load("CZLC/mc_auroc")
    return metric.compute(predictions=list(probs), references=list(golds))["mc_auroc_score"]


def get_czech_news_target(dataset):
    return dataset["category"] - 1


def process_umimeto(dataset):
    """Transform umimeto-qa into MC-friendly form: choices=[A_text, B_text], gold=0/1."""
    def _map(ex):
        ans = ex.get("correct_answer", "A").strip()
        ex["choices"] = [ex.get("A", ""), ex.get("B", "")]
        ex["gold"] = 0 if ans == "A" else 1
        return ex
    return dataset.map(_map)


def process_results_qa_bcm(doc, results):
    """SQuAD-style exact_match + token-F1 over a list of acceptable gold answers."""
    from transformers.data.metrics import squad_metrics

    pred = results[0]
    refs = doc.get("answers", [])
    if isinstance(refs, str):
        refs = [refs]
    if not refs:
        return {"exact_match": 0.0, "f1": 0.0}
    em = max(squad_metrics.compute_exact(r, pred) for r in refs)
    f1 = max(squad_metrics.compute_f1(r, pred) for r in refs)
    return {"exact_match": em, "f1": f1}


def rouge_raw_without_bootstrap(predictions, references, select: Optional[str] = None):
    module = evaluate.load("CZLC/rouge_raw")
    return module.compute(
        predictions=predictions,
        references=references,
        select=select,
        aggregate=False,
    )


def rouge_raw_r2_mid_f_without_bootstrap(predictions, references):
    return rouge_raw_without_bootstrap(predictions, references, "2_fmeasure")


class BCZMTask(ConfigurableTask):
    """Thin ConfigurableTask wrapper for YAML `class: !function utils.BCZMTask`.
    Workaround for lm_eval/tasks/__init__.py::_load_task, which only restores
    `config.task` for sub-tasks that go through the python-task branch (those
    with a `class:` field); without this wrapper, grouped sub-tasks end up with
    config.task=None, failing the duplicate check.
    """

    def __init__(self, config=None, **kwargs):
        if isinstance(config, dict):
            config = {k: v for k, v in config.items() if k != "class"}
        super().__init__(config=config, **kwargs)


class BCZMMultipleChoiceTask(ConfigurableTask):
    """ConfigurableTask variant that exposes BenCzechMark classification metrics
    (acc, macro_f1, avg_mcauroc) from a single loglikelihood pass."""

    def __init__(self, config=None, **kwargs):
        if isinstance(config, dict):
            config = {k: v for k, v in config.items() if k != "class"}
        super().__init__(config=config, **kwargs)

    def process_results(self, doc: dict, results: Iterable[tuple[float, bool]]) -> dict:
        lls = [r[0] for r in results]
        choices = self.doc_to_choice(doc)

        gold = self.doc_to_text(doc) if self.multiple_input else self.doc_to_target(doc)

        if isinstance(gold, list):
            gold = [i if i < len(choices) else -100 for i in gold]
            gold_index_error = -100 in gold
        else:
            if isinstance(gold, int):
                gold = gold if gold < len(choices) else -100
            elif isinstance(gold, str):
                gold = choices.index(gold) if gold in choices else -100
            gold_index_error = gold == -100

        if gold_index_error:
            eval_logger.warning(
                f"Label index out of range of available choices. Sample:\n\n{doc}\n\n"
            )

        lls_arr = np.asarray(lls, dtype=np.float64)
        probs = np.exp(lls_arr - lls_arr.max())
        probs = (probs / probs.sum()).tolist()
        pred = int(np.argmax(lls_arr))

        if self.multiple_target:
            acc = 1.0 if pred in gold else 0.0
        else:
            acc = 1.0 if pred == gold else 0.0

        use_metric = list(self._metric_fn_list.keys())
        return {
            **({"acc": acc} if "acc" in use_metric else {}),
            **({"macro_f1": (gold, pred)} if "macro_f1" in use_metric else {}),
            **({"avg_mcauroc": (gold, probs)} if "avg_mcauroc" in use_metric else {}),
        }

    def higher_is_better(self) -> dict:
        return {"acc": True, "macro_f1": True, "avg_mcauroc": True}

    def aggregation(self) -> dict:
        return {
            "acc": mean,
            "macro_f1": aggregate_macro_f1_score,
            "avg_mcauroc": aggregate_avg_mcauroc,
        }
