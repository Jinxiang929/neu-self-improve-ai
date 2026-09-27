import json
from pathlib import Path
from typing import Any

from openai import OpenAI


class EnvironmentDesigner:
    """
    SPADE-inspired Environment Designer for Healthcare_GYM.

    Instead of generating an entire Gym environment from scratch,
    the designer generates Healthcare_GYM-compatible task
    specifications that are executed by the existing clinical
    diagnosis environment.
    """

    def __init__(
        self,
        tasks_path: str = "data/domains/clinical_diagnosis/tasks.json",
        db_path: str = "data/domains/clinical_diagnosis/db.json",
        model: str = "gpt-5-mini",
    ):
        self.tasks_path = Path(tasks_path)
        self.db_path = Path(db_path)

        self.model = model
        self.client = OpenAI()

        with self.tasks_path.open("r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        with self.db_path.open("r", encoding="utf-8") as f:
            self.db = json.load(f)

    def get_corpus_examples(self, n: int = 3) -> list[dict[str, Any]]:
        """
        Return existing Healthcare_GYM tasks as corpus grounding.
        """
        return self.corpus[:n]

    def get_available_patients(self) -> list[str]:
        """
        Return patient IDs that generated tasks may reference.
        """
        return list(self.db.get("patients", {}).keys())
    
    def build_prompt(
        self,
        iteration: int,
        memory_records: list[dict[str, Any]],
        num_examples: int = 3,
        validation_feedback: list[str] | None = None,
    ) -> str:
        """
        Build a SPADE-inspired prompt for generating one new
        Healthcare_GYM clinical diagnosis task.
        """

        corpus_examples = self.get_corpus_examples(num_examples)

        patients = self.db.get("patients", {})

        # Give the designer enough patient information to create
        # a grounded task without inventing nonexistent patients.
        patient_context = {}

        for patient_id, patient in patients.items():
            patient_context[patient_id] = {
                "name": patient.get("name"),
                "age": patient.get("age"),
                "sex": patient.get("sex"),
                "chief_complaint": patient.get("chief_complaint"),
                "allergies": patient.get("allergies", []),
                "conditions": patient.get("conditions", []),
                "medications": patient.get("medications", []),
                "vital_signs": patient.get("vital_signs", []),
                "lab_results": patient.get("lab_results", []),
                "clinical_notes": patient.get("clinical_notes", []),
            }

        # Only include recent memory so the prompt does not grow forever.
        recent_memory = memory_records[-3:]
        
        previous_complexities = [
            record.get("complexity_score", 0)
            for record in memory_records
            if record.get("valid")
        ]

        previous_max_complexity = (
            max(previous_complexities)
            if previous_complexities
            else 0
        )
        
        feedback_text = (
            "\n".join(f"- {error}" for error in validation_feedback)
            if validation_feedback
            else "None. This is the first generation attempt for this iteration."
        )

        prompt = f"""
    You are the Environment Designer in a SPADE-inspired
    environment generation loop.

    Your job is to create ONE new multi-turn clinical diagnosis
    environment for Healthcare_GYM.

    This is iteration {iteration} of the environment-design loop.

    The environment must be grounded in the existing Healthcare_GYM
    clinical database and must be executable using the existing
    Healthcare_GYM tools.

    IMPORTANT:
    - Do NOT create a new patient.
    - Do NOT invent patient IDs.
    - Select an existing patient from PATIENT DATABASE CONTEXT.
    - Do NOT create new tools.
    - The task must require multiple clinical reasoning/tool-use steps.
    - The task must not be solvable by a single tool call.
    - Required actions must correspond to information or operations
    available in Healthcare_GYM.
    - The generated task should be meaningfully different from the
    corpus examples and previous generated environments.
    - As the iteration number increases, prefer more complex tasks:
    more relevant constraints, more required actions, and longer
    reasoning chains, while keeping the task feasible.
    - Every required action must contribute to solving the clinical task.
    - Do not make the task impossible merely to increase difficulty.
    - Ground every required clinical finding in data that already exists
    in the selected patient's Healthcare_GYM database record.

    - order_lab_test creates an order but does NOT guarantee that a new
    test result will become available during the episode.

    - Do NOT require the agent to observe or reason from a newly ordered
    test result unless that result already exists in the patient's
    database record.

    - Do NOT claim that a diagnosis is confirmed by imaging, laboratory
    testing, or another procedure unless the corresponding result is
    present in the patient database context.

    - A tool being callable does not mean that its requested result exists.
    Keep the task semantically feasible using the available patient data.

    - The highest validated complexity score from previous generated
    environments is {previous_max_complexity}.

    - Generate an environment that is structurally more complex than
    previous validated environments.

    - To increase complexity meaningfully, use a coherent combination of
    additional clinical constraints, required tool interactions,
    evidence integration, and evaluation criteria.

    - Do NOT increase complexity by adding irrelevant or redundant actions.

    - Do NOT sacrifice feasibility merely to obtain a higher complexity.

    PATIENT DATABASE CONTEXT:
    {json.dumps(patient_context, indent=2, ensure_ascii=False)}

    EXISTING HEALTHCARE_GYM TASKS (CORPUS):
    {json.dumps(corpus_examples, indent=2, ensure_ascii=False)}

    PREVIOUS GENERATED ENVIRONMENTS (MEMORY):
    {json.dumps(recent_memory, indent=2, ensure_ascii=False)}
    
    VALIDATION FEEDBACK FROM THE PREVIOUS ATTEMPT:
    {feedback_text}

    If validation feedback is provided above:
    - Correct every reported validation problem.
    - Do not repeat the same invalid assumption or unsupported requirement.
    - Preserve the goal of generating a challenging environment.
    - The corrected environment must remain grounded in the patient database.

    OUTPUT FORMAT:

    Return ONLY one valid JSON object.

    Use exactly this general Healthcare_GYM task structure:

    {{
    "id": "spade_clinical_<unique_name>",
    "description": {{
        "purpose": "...",
        "difficulty": "easy|moderate|hard",
        "key_challenges": [
        "...",
        "..."
        ]
    }},
    "user_scenario": {{
        "persona": "...",
        "instructions": "..."
    }},
    "ticket": "...",
    "evaluation_criteria": {{
        "actions": [
        {{
            "action_id": "...",
            "name": "...",
            "arguments": {{}},
            "info": "..."
        }}
        ],
        "nl_assertions": [
        "...",
        "..."
        ],
        "reward_basis": [
        "ACTION",
        "NL_ASSERTION"
        ]
    }}
    }}

    Design a coherent multi-step clinical workflow whose difficulty
    is appropriate for iteration {iteration}.

    Return JSON only. Do not use Markdown fences.
    """

        return prompt.strip()
    
    
    def generate(
        self,
        iteration: int,
        memory_records: list[dict[str, Any]],
        validation_feedback: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate one Healthcare_GYM-compatible clinical task.
        """

        prompt = self.build_prompt(
            iteration=iteration,
            memory_records=memory_records,
            validation_feedback=validation_feedback,
        )

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert environment designer. "
                        "Generate grounded, multi-step tool-use environments "
                        "that strictly follow the requested JSON schema."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        output_text = response.output_text.strip()

        # Be tolerant if the model unexpectedly adds Markdown fences.
        if output_text.startswith("```json"):
            output_text = output_text[7:]

        if output_text.startswith("```"):
            output_text = output_text[3:]

        if output_text.endswith("```"):
            output_text = output_text[:-3]

        output_text = output_text.strip()

        try:
            task = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Environment Designer returned invalid JSON:\n"
                f"{output_text}"
            ) from exc

        if not isinstance(task, dict):
            raise ValueError(
                "Environment Designer output must be one JSON object."
            )
            
        output_dir = Path("spade_designer/generated")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"candidate_{iteration:03d}.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(task, f, indent=2, ensure_ascii=False)

        print(f"Saved generated environment to: {output_path}")

        return task