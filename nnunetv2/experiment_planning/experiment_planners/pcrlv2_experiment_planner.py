from typing import Union, List, Tuple

from nnunetv2.experiment_planning.experiment_planners.pretraining_experiment_planner import PretrainingExperimentPlanner


class PCRLv2ExperimentPlanner(PretrainingExperimentPlanner):
    def __init__(
            self,
            dataset_name_or_id: Union[str, int],
            gpu_memory_target_in_gb: float = 8,
            preprocessor_name: str = "DefaultPreprocessor",
            plans_name: str = "nnUNetPreprocessPlans",
            overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
            suppress_transpose: bool = False,
    ):
        super().__init__(
            dataset_name_or_id=dataset_name_or_id,
            gpu_memory_target_in_gb=gpu_memory_target_in_gb,
            preprocessor_name="PCRLv2Preprocessor",
            plans_name="nnUNetPreprocessPlansPCRLv2",
            overwrite_target_spacing=overwrite_target_spacing,
            suppress_transpose=suppress_transpose,
        )
