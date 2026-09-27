import json
from pathlib import Path
from typing import Any
import gymnasium as gym
from bioagents.gym.agent_env import register_bioagent_gym


class TaskValidator:
    """
    Validates generated Healthcare_GYM clinical diagnosis tasks.
    """

    REQUIRED_FIELDS = {
        "id",
        "description",
        "ticket",
        "evaluation_criteria",
    }

    REQUIRED_EVALUATION_FIELDS = {
        "actions",
        "reward_basis",
    }

    def __init__(
        self,
        db_path: str = "data/domains/clinical_diagnosis/db.json",
    ):
        self.db_path = Path(db_path)

        with self.db_path.open("r", encoding="utf-8") as f:
            self.db = json.load(f)

        self.patient_ids = set(self.db.get("patients", {}).keys())

    def validate_schema(self, task: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Check whether a generated task has the minimum structure
        required by Healthcare_GYM.
        """

        errors = []

        # 1. Check top-level fields
        missing_fields = self.REQUIRED_FIELDS - set(task.keys())

        if missing_fields:
            errors.append(
                f"Missing required fields: {sorted(missing_fields)}"
            )

        # Cannot safely continue without evaluation criteria
        evaluation = task.get("evaluation_criteria")

        if not isinstance(evaluation, dict):
            errors.append(
                "evaluation_criteria must be a dictionary"
            )
            return False, errors

        # 2. Check evaluation fields
        missing_eval_fields = (
            self.REQUIRED_EVALUATION_FIELDS
            - set(evaluation.keys())
        )

        if missing_eval_fields:
            errors.append(
                "Missing evaluation fields: "
                f"{sorted(missing_eval_fields)}"
            )

        # 3. Check actions
        actions = evaluation.get("actions", [])

        if not isinstance(actions, list):
            errors.append("actions must be a list")
        else:
            for index, action in enumerate(actions):
                if not isinstance(action, dict):
                    errors.append(
                        f"Action {index} must be a dictionary"
                    )
                    continue

                if "name" not in action:
                    errors.append(
                        f"Action {index} is missing 'name'"
                    )

                if "arguments" not in action:
                    errors.append(
                        f"Action {index} is missing 'arguments'"
                    )

        # 4. Check patient references
        referenced_patients = self._find_patient_ids(task)

        for patient_id in referenced_patients:
            if patient_id not in self.patient_ids:
                errors.append(
                    f"Unknown patient ID: {patient_id}"
                )

        return len(errors) == 0, errors

    def validate_executable(
        self,
        task: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """
        Smoke-test a generated task inside Healthcare_GYM.

        Validation includes:
        1. Schema validation
        2. Task injection through reset()
        3. Execution of every required tool action
        """

        errors = []

        schema_valid, schema_errors = self.validate_schema(task)

        if not schema_valid:
            return False, schema_errors

        register_bioagent_gym()

        env = None

        try:
            env = gym.make(
                "BioAgent-v0",
                domain="clinical_diagnosis",
            )

            observation, info = env.reset(
                options={"task_data": task}
            )

            if not isinstance(observation, str):
                errors.append(
                    "reset() did not return a string observation"
                )

            if info.get("task_id") != task.get("id"):
                errors.append(
                    "Injected task ID does not match environment task ID"
                )

            actions = task.get(
                "evaluation_criteria", {}
            ).get("actions", [])

            for index, action in enumerate(actions):
                tool_name = action.get("name")
                arguments = action.get("arguments", {})

                tool_call = json.dumps({
                    "name": tool_name,
                    "arguments": arguments,
                })

                try:
                    (
                        next_observation,
                        reward,
                        terminated,
                        truncated,
                        step_info,
                    ) = env.step(tool_call)

                except Exception as exc:
                    errors.append(
                        f"Action {index} '{tool_name}' crashed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue

                tool_response = step_info.get("tool_response")

                if tool_response is None:
                    errors.append(
                        f"Action {index} '{tool_name}' "
                        "was not recognized as a tool call"
                    )
                    continue

                if tool_response.get("error"):
                    errors.append(
                        f"Action {index} '{tool_name}' failed: "
                        f"{tool_response.get('content')}"
                    )

        except Exception as exc:
            errors.append(
                "Environment execution failed: "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if env is not None:
                env.close()

        return len(errors) == 0, errors

    def _find_patient_ids(self, task: dict[str, Any]) -> set[str]:
        """
        Find patient IDs referenced in evaluation actions.
        """

        patient_ids = set()

        evaluation = task.get("evaluation_criteria", {})

        for action in evaluation.get("actions", []):
            if not isinstance(action, dict):
                continue

            arguments = action.get("arguments", {})

            if not isinstance(arguments, dict):
                continue

            patient_id = arguments.get("patient_id")

            if patient_id:
                patient_ids.add(patient_id)

        return patient_ids
    
    def calculate_complexity(
        self,
        task: dict[str, Any],
    ) -> float:
        """
        Calculate a simple structural complexity score for a generated task.

        The score is based on:
        - number of required tool actions
        - number of natural-language assertions
        - number of stated key challenges
        - declared difficulty
        """

        evaluation = task.get("evaluation_criteria", {})
        description = task.get("description", {})

        num_actions = len(
            evaluation.get("actions", [])
        )

        num_assertions = len(
            evaluation.get("nl_assertions", [])
        )

        num_challenges = len(
            description.get("key_challenges", [])
        )

        difficulty = description.get(
            "difficulty",
            "moderate",
        ).lower()

        difficulty_weight = {
            "easy": 1,
            "moderate": 2,
            "hard": 3,
        }.get(difficulty, 2)

        score = (
            num_actions
            + 0.5 * num_assertions
            + 0.5 * num_challenges
            + difficulty_weight
        )

        return round(score, 2)