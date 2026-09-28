# main/evaluation/__init__.py
from .ornament_benchmark_dataset import (
    OrnamentToBackboneBenchmarkDataset,
    make_ornament_benchmark_dataloader,
)
from .ornament_to_backbone_eval import (
    OrnamentToBackboneEvalConfig,
    evaluate_ornament_to_backbone,
)
from .score_extraction import (
    BackbonePrediction,
    PointerModelAdapter,
    build_pointer_adapter_from_model,
)

from .gttm_benchmark_dataset import make_gttm_benchmark_dataloader
from .gttm_eval import GTTMBackboneEvalConfig, evaluate_gttm_backbone
from .score_extraction import ScoreArrayAdapter

__all__ = [
    "OrnamentToBackboneBenchmarkDataset",
    "make_ornament_benchmark_dataloader",
    "OrnamentToBackboneEvalConfig",
    "evaluate_ornament_to_backbone",
    "BackbonePrediction",
    "PointerModelAdapter",
    "build_pointer_adapter_from_model",
    "make_gttm_benchmark_dataloader",
    "GTTMBackboneEvalConfig",
    "evaluate_gttm_backbone",
    "ScoreArrayAdapter",
]