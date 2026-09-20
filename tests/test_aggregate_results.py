from scripts.aggregate_results import (
    build_summary,
    deduplicate,
    experiment_labels,
)


def _evaluation(experiment, nfe, fid, sample_count=None):
    return {
        "experiment": experiment,
        "algorithm": experiment,
        "algorithm_class": "ExampleAlgorithm",
        "seed": 0,
        "record_type": "evaluation",
        "epoch": None,
        "nfe": nfe,
        "fid": fid,
        "is_mean": 1.0,
        "num_generated_samples": sample_count,
    }


def test_evaluation_deduplication_keeps_last_occurrence():
    records = [
        _evaluation("consistency_cifar10", 2, 434.0),
        _evaluation("consistency_cifar10", 2, 119.46, 5000),
    ]

    deduplicated = deduplicate(records)

    assert len(deduplicated) == 1
    assert deduplicated[0]["fid"] == 119.46
    assert deduplicated[0]["num_generated_samples"] == 5000


def test_summary_exposes_sample_count_and_full_nfe_grid():
    records = [
        _evaluation("fm_cifar10_5k", 1, 374.0, 5000),
        _evaluation("fm_cifar10_5k", 2, 200.0, 5000),
        _evaluation("fm_cifar10_5k", 5, 63.6, 5000),
        _evaluation("fm_cifar10_5k", 10, 47.1, 5000),
        _evaluation("fm_cifar10_5k", 20, 41.8, 5000),
    ]

    summary = build_summary(records)[0]

    assert summary["num_generated_samples"] == 5000
    assert [summary[f"fid_at_{nfe}"] for nfe in (1, 2, 5, 10, 20)] == [
        374.0,
        200.0,
        63.6,
        47.1,
        41.8,
    ]


def test_legacy_null_sample_count_is_reported_as_unknown():
    summary = build_summary([_evaluation("fm_cifar10", 5, 63.6)])[0]
    assert summary["num_generated_samples"] == "unknown"


def test_plot_membership_and_labels_come_from_record_metadata():
    records = [
        {
            "experiment": "fm_lognorm_cifar10",
            "experiment_name": "fm_lognorm",
            "dataset": "cifar10",
            "algorithm_class": "FlowMatchingLognormAlgorithm",
        },
        {
            "experiment": "mf_v3_exact_jvp_b128_cifar10",
            "experiment_name": "mf_v3_exact_jvp_b128",
            "dataset": "cifar10",
            "algorithm_class": "MeanFlowAlgorithm",
        },
        {
            "experiment": "fm_celeba",
            "experiment_name": "fm",
            "dataset": "celeba",
            "algorithm_class": "FlowMatchingAlgorithm",
        },
    ]

    assert experiment_labels(records, "cifar10") == {
        "fm_lognorm_cifar10": "FM-LN",
        "mf_v3_exact_jvp_b128_cifar10": "Mean Flow [mf_v3_exact_jvp_b128]",
    }
    assert experiment_labels(records, "celeba") == {"fm_celeba": "FM"}
