from spade_designer.designer import EnvironmentDesigner
from spade_designer.memory import EnvironmentMemory
from spade_designer.validator import TaskValidator


def run_spade_loop(
    start_iteration: int = 1,
    num_iterations: int = 3,
) -> None:
    """
    Run the SPADE-inspired Environment Designer loop.

    Each iteration:
    1. Generates a new Healthcare_GYM task.
    2. Validates its schema.
    3. Validates executability in Healthcare_GYM.
    4. Calculates structural complexity.
    5. Stores the result in environment memory.
    """

    designer = EnvironmentDesigner()
    validator = TaskValidator()
    memory = EnvironmentMemory()

    print("=" * 70)
    print("SPADE-inspired Healthcare_GYM Environment Designer")
    print("=" * 70)

    for iteration in range(
        start_iteration,
        start_iteration + num_iterations,
    ):
        print(f"\nIteration {iteration}")
        print("-" * 70)

        max_attempts = 3
        task = None
        valid = False
        validation_feedback = None

        for attempt in range(1, max_attempts + 1):
            print(
                f"\nGeneration attempt "
                f"{attempt}/{max_attempts}"
            )

            try:
                task = designer.generate(
                    iteration=iteration,
                    memory_records=memory.records,
                    validation_feedback=validation_feedback,
                )

            except Exception as exc:
                print(
                    "Generation failed:",
                    f"{type(exc).__name__}: {exc}",
                )
                continue

            print("Generated:", task.get("id"))

            schema_valid, schema_errors = (
                validator.validate_schema(task)
            )

            print("Schema valid:", schema_valid)

            if not schema_valid:
                for error in schema_errors:
                    print("  -", error)

                validation_feedback = schema_errors
                continue

            executable_valid, executable_errors = (
                validator.validate_executable(task)
            )

            print(
                "Executable valid:",
                executable_valid,
            )

            if not executable_valid:
                for error in executable_errors:
                    print("  -", error)

                validation_feedback = executable_errors
                continue

            valid = True
            break


        if task is None:
            print(
                f"Iteration {iteration} failed: "
                "no task was generated."
            )
            continue

        if not valid:
            print(
                f"Iteration {iteration} failed validation "
                f"after {max_attempts} attempts."
            )
            print("Not stored in environment memory.")
            continue


        complexity_score = (
            validator.calculate_complexity(task)
        )

        print(
            "Complexity score:",
            complexity_score,
        )

        memory.add(
            iteration=iteration,
            task=task,
            complexity_score=complexity_score,
            valid=valid,
        )

        print(
            "Stored in memory:",
            "VALID" if valid else "INVALID",
        )

    print("\n" + "=" * 70)
    print("Loop complete")
    print("=" * 70)

    print("\nEnvironment memory:")

    for record in memory.records:
        print(
            f"Iteration {record['iteration']}: "
            f"{record['task_id']} | "
            f"difficulty={record['difficulty']} | "
            f"complexity={record['complexity_score']} | "
            f"valid={record['valid']}"
        )


if __name__ == "__main__":
    run_spade_loop()