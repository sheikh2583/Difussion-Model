from scripts.aggregate_results import (
    _dataset_identity,
    add_pareto_flags,
    build_algorithm_progression,
    build_representation_comparison,
    build_summary,
    build_transfer_consistency,
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


def _point(dataset, representation, method, fid, nfe=5):
    return {
        "experiment": f"{method}_{dataset}_{representation}",
        "comparison": f"{method}_{dataset}_{representation}@machine@source",
        "dataset": dataset if representation == "pixel" else f"{dataset}_latent",
        "dataset_family": dataset,
        "representation": representation,
        "method_key": method,
        "method_order": 0 if method == "fm" else 1,
        "algorithm_class": "ExampleAlgorithm",
        "epoch": 100,
        "nfe": nfe,
        "seed": 0,
        "num_generated_samples": 5000,
        "fid": fid,
        "is_mean": 2.0,
        "machine_label": "machine",
        "code_identity": "source",
        "backbone_signature": "backbone",
        "state_values": 3072 if representation == "pixel" else 768,
        "sampling_time_seconds": 10.0 if representation == "pixel" else 2.0,
        "codec_checkpoint": "codec.pt" if representation == "latent" else None,
    }


def test_dataset_identity_separates_dataset_from_representation():
    assert _dataset_identity("cifar10") == ("cifar10", "pixel")
    assert _dataset_identity("cifar10_latent") == ("cifar10", "latent")


def test_progression_and_cross_dataset_transfer_are_fm_relative():
    points = [
        _point("cifar10", "pixel", "fm", 20.0),
        _point("cifar10", "pixel", "fm_lognorm", 15.0),
        _point("celeba", "pixel", "fm", 40.0),
        _point("celeba", "pixel", "fm_lognorm", 30.0),
    ]
    progression = build_algorithm_progression(points)
    gains = [
        row for row in progression if row["method_key"] == "fm_lognorm"
    ]
    assert [row["fid_improvement_percent"] for row in gains] == [25.0, 25.0]

    transfer = build_transfer_consistency(progression)
    assert transfer[0]["dataset_count"] == 2
    assert transfer[0]["transfer_status"] == "consistent_improvement"


def test_progression_rejects_incompatible_backbone():
    baseline = _point("cifar10", "pixel", "fm", 20.0)
    candidate = _point("cifar10", "pixel", "fm_lognorm", 15.0)
    candidate["backbone_signature"] = "different"
    row = build_algorithm_progression([baseline, candidate])[1]
    assert row["comparison_status"] == "no_protocol_compatible_fm"
    assert row["fid_improvement_percent"] is None


def test_representation_pair_reports_codec_cost_and_state_reduction():
    pixel = _point("celeba", "pixel", "fm", 20.0)
    latent = _point("celeba", "latent", "fm", 22.0)
    pair = build_representation_comparison([pixel, latent])[0]
    assert pair["latent_minus_pixel_fid"] == 2.0
    assert pair["sampling_speedup_pixel_over_latent"] == 5.0
    assert pair["state_reduction_factor"] == 4.0
    assert "codec reconstruction error" in pair["interpretation_warning"]


def test_pareto_flags_quality_against_nfe():
    fast_bad = _point("cifar10", "pixel", "fm", 30.0, nfe=1)
    slow_good = _point("cifar10", "pixel", "fm_lognorm", 20.0, nfe=5)
    dominated = _point("cifar10", "pixel", "mf", 35.0, nfe=5)
    rows = add_pareto_flags([fast_bad, slow_good, dominated])
    flags = {row["method_key"]: row["pareto_fid_vs_nfe"] for row in rows}
    assert flags == {"fm": True, "fm_lognorm": True, "mf": False}
