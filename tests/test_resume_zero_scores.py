import importlib
import json

import pytest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.hillclimb import HillClimb, HillClimbProfiler
from llm4ad.method.hillclimb.resume import resume_hillclimb
from llm4ad.tools.profiler import ProfilerBase


def write_samples(path, scores):
    samples_dir = path / "samples"
    samples_dir.mkdir()
    samples = [
        {
            "function": f"def heuristic(x):\n    return x + {index}\n",
            "score": score,
            "algorithm": f"algorithm {index}",
            "sample_order": index + 1,
        }
        for index, score in enumerate(scores)
    ]
    (samples_dir / "samples_1~200.json").write_text(json.dumps(samples))
    return samples


@pytest.mark.parametrize(
    "method",
    [
        "eoh",
        "reevo",
        "mcts_ahd",
        "funsearch",
        "hillclimb",
        "randsample",
        pytest.param(
            "lhns",
            marks=pytest.mark.xfail(
                raises=ImportError,
                strict=True,
                reason="Existing LHNS resume imports EoH from lhns.lhns",
            ),
        ),
    ],
)
def test_resume_loader_preserves_zero_scores(tmp_path, method):
    samples = write_samples(tmp_path, [0, 0.0, -2, 3, None])
    resume = importlib.import_module(f"llm4ad.method.{method}.resume")
    funcs, scores, order, algorithms = resume._get_all_samples_and_scores(
        str(tmp_path), get_algorithm=True
    )
    assert scores == [0, 0.0, -2, 3, float("-inf")]
    assert funcs == [sample["function"] for sample in samples]
    assert algorithms == [sample["algorithm"] for sample in samples]
    assert order == 5


@pytest.mark.parametrize("valid_only", [False, True])
def test_logfile_preserves_zero_and_filters_only_invalid_scores(tmp_path, valid_only):
    samples = write_samples(tmp_path, [0, 0.0, -2, 3, None])
    funcs, scores = ProfilerBase.load_logfile(str(tmp_path), valid_only=valid_only)
    expected = [0, 0.0, -2, 3] if valid_only else [0, 0.0, -2, 3, float("-inf")]
    assert scores == expected
    assert funcs == [sample["function"] for sample in samples[: len(expected)]]


class ResumeOnlyEvaluation(Evaluation):
    def evaluate_program(self, program_str, callable_func, **kwargs):
        raise AssertionError("Resuming must not evaluate or contact an LLM")


def test_hillclimb_resume_keeps_zero_as_best_score(tmp_path):
    samples = write_samples(tmp_path, [-2, 0])
    sample_file = tmp_path / "samples" / "samples_1~200.json"
    before = sample_file.read_bytes()
    evaluation = ResumeOnlyEvaluation(samples[0]["function"], safe_evaluate=False)
    method = HillClimb(LLM(), evaluation, num_samplers=1, num_evaluators=1)
    profiler = HillClimbProfiler(log_dir=str(tmp_path), create_random_path=False)
    method._profiler = profiler
    try:
        resume_hillclimb(method)
        assert method._best_function_found.body.strip() == "return x + 1"
        assert profiler._cur_best_program_score == 0
        assert method._tot_sample_nums == 2
        assert sample_file.read_bytes() == before
    finally:
        method._evaluation_executor.shutdown()
