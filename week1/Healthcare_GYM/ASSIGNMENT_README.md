# SPADE-Inspired Environment Designer for Healthcare_GYM

## 1. What I Wanted to Build

For this assignment, I wanted to explore how the Environment Designer
idea from SPADE could be applied to an existing domain-specific
environment library.

I chose **Healthcare_GYM**, a healthcare agent environment built around
clinical tasks and tool use. The repository is actively maintained, with
commits within the last six months.

The goal of this project was **not to train a policy**. Instead, I
focused specifically on the Environment Designer side of SPADE:

> Can an LLM progressively generate new, executable, and increasingly
> complex clinical environments while reusing the existing
> Healthcare_GYM runtime?

My final implementation generates Healthcare_GYM-compatible clinical
tasks, validates them by actually running them inside the environment,
measures their structural complexity, stores successful environments in
memory, and uses that memory to guide later generations.

The final loop looks like this:

``` text
Healthcare_GYM task corpus
            +
Healthcare_GYM patient database
            +
Previous generated environments
            |
            v
     Environment Designer
            |
            v
     Generated task JSON
            |
            v
   Schema validation
            |
            v
Executable validation in Healthcare_GYM
            |
            v
     Complexity score
            |
            v
      Environment memory
            |
            +--------------------+
                                 |
                                 v
                         Next iteration
```

This README explains how I arrived at this design step by step.

## 2. Why Healthcare_GYM?

The assignment required an actively developed environment library in a
specific domain.

I chose Healthcare_GYM because it already provides a fairly
sophisticated simulated healthcare environment rather than a simple toy
state space.

The repository contains multiple healthcare domains, including a
clinical diagnosis environment. It also provides:

-   patient state stored in a simulated database,
-   clinical tools,
-   task definitions,
-   evaluation criteria,
-   Gym-compatible `reset()` and `step()` interfaces,
-   and agent-oriented multi-step workflows.

The GitHub repository also showed recent development activity, including
commits from the previous month when I worked on this assignment, so it
satisfied the active-development requirement.

I decided to use the `clinical_diagnosis` domain because it naturally
supports multi-step environments. A task may require retrieving patient
information, checking vital signs and laboratory results, reasoning over
a differential diagnosis, searching clinical guidelines, ordering tests,
recording a diagnosis, and transferring the patient to a specialist.

That seemed much closer to the multi-turn environment design problem
described by SPADE than a simple one-step classification task.

## 3. First Step: Can I Run Healthcare_GYM at All?

Before thinking about SPADE, I first needed to understand the library
itself.

I installed the repository locally:

``` bash
pip install -e .
```

Then I wrote a minimal smoke test in:

``` text
test_env.py
```

The initial test was:

``` python
import gymnasium as gym
from bioagents.gym.agent_env import register_bioagent_gym

register_bioagent_gym()

env = gym.make(
    "BioAgent-v0",
    domain="clinical_diagnosis",
    task_id="dx_pneumonia_001",
)

obs, info = env.reset()

print(obs)
print(info)
```

This successfully created the environment and returned the initial
clinical task.

This test was important because it answered my first question:

> Does Healthcare_GYM already provide the executable environment layer,
> or do I need to build that myself?

The answer was that Healthcare_GYM already provides it.

That changed how I thought about implementing SPADE.

## 4. Understanding `step()`

My next question was how an agent actually interacts with the
environment.

At first, I tried passing a Python dictionary directly to:

``` python
env.step(...)
```

That did not work.

I then inspected the Healthcare_GYM implementation and found that the
environment expects the action to be passed as a **string**,
representing a tool call.

For example:

``` python
action = (
    '{"name": "get_patient_info", '
    '"arguments": {"patient_id": "P001"}}'
)

obs, reward, terminated, truncated, info = env.step(action)
```

This worked.

That small experiment was useful because it showed that Healthcare_GYM
already has a tool-dispatch layer.

The environment is not just:

``` text
observation -> arbitrary text -> reward
```

It already understands structured actions such as:

``` text
get_patient_info
get_vital_signs
get_lab_results
order_lab_test
search_clinical_guidelines
record_diagnosis
transfer_to_specialist
```

At this point, I realized that generating an entirely new Gym class for
every environment would duplicate a lot of functionality that
Healthcare_GYM already provides.

## 5. Finding the Actual Environment Schema

The next step was understanding where a clinical environment comes from.

I inspected the files under:

``` text
data/domains/clinical_diagnosis/
```

The most important files were:

``` text
db.json
tasks.json
tasks_auto_generated.json
tasks_scaled.json
policy.md
```

I also inspected the clinical diagnosis implementation, including the
environment, tools, and data model.

This led to an important distinction.

### `db.json`

This represents the simulated clinical world.

It contains information such as:

-   patients,
-   demographics,
-   allergies,
-   medications,
-   conditions,
-   vital signs,
-   laboratory results,
-   clinical notes,
-   drug interactions,
-   and clinical guidelines.

### `tasks.json`

This defines what the agent is expected to accomplish.

A task contains fields such as:

``` json
{
  "id": "...",
  "description": {
    "purpose": "...",
    "difficulty": "...",
    "key_challenges": []
  },
  "user_scenario": {
    "persona": "...",
    "instructions": "..."
  },
  "ticket": "...",
  "evaluation_criteria": {
    "actions": [],
    "nl_assertions": [],
    "reward_basis": []
  }
}
```

This was the key discovery for the project.

Healthcare_GYM separates:

``` text
World state          -> db.json
Task specification   -> tasks.json
Tools                -> tools.py
Runtime              -> BioAgentGymEnv
```

So the task JSON is effectively the part that tells the existing
environment **what problem should be solved**.

## 6. How This Changed My Interpretation of SPADE

In SPADE, the Environment Designer generates executable environments.

For the tool-use setting, the generated environment defines things such
as tools, backend state, user messages, state transitions, completion
criteria, and multi-turn interaction behavior.

My first thought was therefore to generate a new Python Gym environment
for every iteration.

After studying Healthcare_GYM, I decided that would not be the right
adaptation.

Healthcare_GYM already owns:

``` text
tools
backend state
Gym reset()/step()
tool parsing
tool dispatch
task evaluation
```

Reimplementing all of those inside generated Python classes would mostly
reproduce functionality already present in the library.

Instead, I treated the Healthcare_GYM task specification as the
**library-specific environment definition**.

My adaptation therefore became:

``` text
Original SPADE

Environment Designer
        |
        v
Executable Python environment
        |
        v
tools + state + reset()/step()


My Healthcare_GYM adaptation

Environment Designer
        |
        v
Healthcare_GYM task JSON
        |
        v
existing tools + db.json + BioAgentGymEnv
        |
        v
reset()/step()
```

This is why I describe the implementation as **SPADE-inspired** rather
than an exact reproduction of SPADE.

## 7. Testing Whether Generated Tasks Can Be Injected

Before building an LLM designer, I needed to know whether a generated
task could actually be inserted into the existing environment.

While reading `agent_env.py`, I found that `reset()` can accept task
data through its options.

That meant I did not have to modify the original `tasks.json` every time
I generated a new environment.

Instead, a generated task could be loaded dynamically and tested
directly against Healthcare_GYM.

This was important because it made the design loop possible:

``` text
generate task
    |
    v
inject task
    |
    v
env.reset(...)
    |
    v
execute required tool calls
```

At this point, I had enough understanding of Healthcare_GYM to start
implementing the Environment Designer.

## 8. Building `designer.py`

The first main component I wrote was:

``` text
spade_designer/designer.py
```

Its responsibility is to ask an LLM to generate one new
Healthcare_GYM-compatible environment.

I used the OpenAI API with:

``` text
gpt-5-mini
```

The first version of the Designer was grounded in two things:

``` text
Existing Healthcare_GYM tasks
+
Current iteration number
```

I used existing tasks as a small corpus so that the model could see what
valid Healthcare_GYM tasks look like.

The generated output had to follow the same general structure as the
existing task schema.

I also added restrictions such as:

``` text
Do not create new patients.
Do not invent patient IDs.
Do not create new tools.
Use existing Healthcare_GYM operations.
Require multiple reasoning/tool-use steps.
Do not make the task solvable by one tool call.
```

This gave me the first basic version of an Environment Designer.

## 9. Problem: Schema-Correct Does Not Mean Clinically Grounded

The next problem appeared when I looked more carefully at generated
environments.

An LLM can produce JSON that looks perfectly reasonable while requiring
information that the simulated environment cannot actually provide.

For example, Healthcare_GYM has:

``` text
order_lab_test
```

It is tempting for the Designer to generate a workflow like:

``` text
1. Order a new test.
2. Read the new result.
3. Make a decision based on that result.
```

But after inspecting the environment, I found that ordering a test does
**not automatically create a new result** during the episode.

That meant the generated task could be syntactically correct but
semantically impossible.

This changed the Designer prompt.

## 10. Adding Database Grounding

To solve that problem, I added patient database context to the prompt.

For every available patient, the Designer can see relevant existing
information such as:

``` text
name
age
sex
chief complaint
allergies
conditions
medications
vital signs
lab results
clinical notes
```

I then added explicit rules:

``` text
Ground every required clinical finding in existing patient data.

order_lab_test creates an order but does not guarantee that
a new result becomes available.

Do not require reasoning from a newly ordered result unless
that result already exists in the database.

Do not claim that a diagnosis is confirmed by imaging,
laboratory testing, or another procedure unless that result
already exists in the patient context.
```

This was one of the most important changes in the project.

The Designer was no longer generating tasks from only a schema.

It was now generating tasks from:

``` text
Task corpus + actual simulated world state
```

## 11. Example: The Pulmonary Embolism Environment

One early generated environment used patient `P004` and focused on
pulmonary embolism.

Initially, I was concerned that the environment might be requiring a CT
result that did not exist.

I went back to `db.json` and inspected the full patient record.

The patient's clinical note already contained a CT pulmonary angiography
result describing bilateral pulmonary emboli and right-heart strain
information.

That meant the generated task could legitimately use this information.

This was a useful lesson during implementation:

> A callable tool does not prove that the corresponding information
> exists. The Designer has to reason from the actual state represented
> by the database.

That observation directly motivated the stronger grounding rules in
`designer.py`.

## 12. Building `validator.py`

Once the Designer could generate tasks, I needed a way to determine
whether a generated environment was actually usable.

I created:

``` text
spade_designer/validator.py
```

I separated validation into two levels.

### Schema validation

The first level checks the structure of the generated task.

For example:

``` text
Does it have an id?
Does it have description?
Does it have a ticket?
Does evaluation_criteria exist?
Are actions structured correctly?
Are referenced patient IDs valid?
```

This catches malformed Designer output before trying to run it.

### Executable validation

The second level actually creates a Healthcare_GYM environment and
injects the generated task.

Then it executes the required actions through the real environment.

Conceptually:

``` text
generated task
      |
      v
gym.make("BioAgent-v0")
      |
      v
reset(task_data=generated_task)
      |
      v
execute evaluation actions
      |
      v
check for runtime/tool errors
```

This was important because a task can pass JSON/schema validation and
still fail when executed.

## 13. Why I Validate by Running the Environment

This part was directly influenced by SPADE.

SPADE does not simply trust generated environment code. Candidate
environments are checked before they are used.

I wanted the same property here.

So instead of treating:

``` text
LLM returned valid JSON
```

as success, I defined success as:

``` text
LLM returned valid JSON
+
Healthcare_GYM accepted the task
+
reset() worked
+
required actions could execute
```

This makes the Designer output much more meaningful than simply saving
generated text.

## 14. Building Environment Memory

The next component was:

``` text
spade_designer/memory.py
```

SPADE uses accumulated environment experience to influence later
environment generation.

For this assignment, I was not training a policy, so I could not
reproduce SPADE's policy-performance-based feedback exactly.

Instead, I created a lightweight Environment Memory containing generated
tasks and information such as:

``` text
iteration
task ID
difficulty
complexity score
validation status
task
```

The Designer receives recent successful environments in its next prompt.

This allows it to see:

``` text
What has already been generated?
What difficulty has already been reached?
What types of tasks should not simply be repeated?
```

## 15. Measuring Complexity

Because this assignment focuses on Environment Designer rather than
policy training, I needed a simple way to observe whether environments
were becoming structurally more difficult.

I implemented a deterministic complexity score based on:

``` text
number of required actions
number of natural-language assertions
number of key challenges
declared difficulty
```

The current score is:

``` python
score = (
    num_actions
    + 0.5 * num_assertions
    + 0.5 * num_challenges
    + difficulty_weight
)
```

where:

``` text
easy     = 1
moderate = 2
hard     = 3
```

This score is **not part of the original SPADE algorithm**.

It is an assignment-specific measurement that lets me inspect whether my
Environment Designer is actually producing progressively more involved
workflows.

I also deliberately avoided asking the model to optimize the exact
formula, because that would make it easy to inflate the score by adding
meaningless actions.

## 16. First Environment Generation Experiments

The first generated environment was:

``` text
spade_clinical_pulmonary_embolism_rivera
```

It passed both:

``` text
Schema validation:      True
Executable validation:  True
```

Its complexity score was:

``` text
17.5
```

The second generated environment was:

``` text
spade_clinical_submassive_pe_p004
```

It also passed both validators.

Its complexity score increased to:

``` text
18.5
```

The second task required more detailed risk stratification and
management reasoning rather than only identifying pulmonary embolism.

At this point, the progression looked promising.

## 17. Iteration 3 Exposed a Problem

The third generated environment changed to a DKA case:

``` text
spade_clinical_dka_marcus
```

This was useful because the Designer did not simply keep generating
pulmonary embolism variants.

It was also valid and executable.

However, the complexity score was:

``` text
18.5
```

The progression was now:

``` text
Iteration 1    17.5
Iteration 2    18.5
Iteration 3    18.5
```

This exposed a weakness in my first curriculum design.

The prompt said that later iterations should **prefer more complex
tasks**, but this was only a soft instruction.

The Designer knew the iteration number and previous tasks, but it did
not have a clear signal showing the highest complexity already achieved.

So the curriculum plateaued.

I kept this result rather than regenerating Iteration 3, because it
showed a real limitation of the first design.

## 18. Making Complexity Feedback Explicit

To address the plateau, I changed the Designer.

From Environment Memory, I calculated:

``` text
highest validated previous complexity
```

and included that information directly in the next prompt.

For Iteration 4, the Designer therefore knew something equivalent to:

``` text
The highest validated complexity score from previous
generated environments is 18.5.

Generate an environment that is structurally more complex
than previous validated environments.
```

I also added an important constraint:

``` text
Do not increase complexity by adding irrelevant or redundant actions.
```

This was necessary because otherwise the model could increase the metric
simply by generating many unnecessary tool calls.

The goal was not a bigger number.

The goal was a more involved but still coherent environment.

## 19. Iteration 4: Did the Feedback Work?

Iteration 4 generated:

``` text
spade_clinical_urgent_upper_gi_bleed_p006
```

The result was:

``` text
Schema valid:       True
Executable valid:   True
Complexity:         23.5
```

This was a significant increase from `18.5`.

I then inspected why the score increased.

The task contained:

``` text
13 required actions
10 NL assertions
5 key challenges
hard difficulty
```

More importantly, the additional actions were not simply duplicates.

The workflow involved hemodynamic assessment, existing laboratory
evidence, serial monitoring, blood preparation, ECG, gastroenterology,
cardiology, blood bank coordination, critical care, diagnosis
documentation, transfusion considerations, antiplatelet management, and
disposition planning.

So the higher score reflected a genuinely longer and more constrained
workflow.

That gave me evidence that explicit Environment Memory feedback was
improving the curriculum rather than only inflating the metric.

## 20. Adding a Generation Retry Loop

The next problem was robustness.

The first implementation effectively did:

``` text
generate
   |
   v
validate
   |
   +-- valid   -> continue
   |
   +-- invalid -> iteration fails
```

But an Environment Designer should be able to recover from generation
errors.

I changed `run_loop.py` so that each iteration can make up to three
generation attempts.

The new flow became:

``` text
Attempt 1
   |
   v
validate
   |
   +-- valid ----------------------+
   |                               |
   +-- invalid                     |
          |                        |
          v                        |
      Attempt 2                    |
          |                        |
          v                        |
       validate                    |
          |                        |
          +-- invalid -> Attempt 3 |
          |                        |
          +-- valid ---------------+
                                   |
                                   v
                                Memory
```

At first, however, this was only a **blind retry**.

The Designer did not know why its previous environment had failed.

## 21. Adding Validation Feedback

I then connected Validator errors back to the Environment Designer.

`designer.py` now accepts:

``` python
validation_feedback
```

If a previous attempt fails, errors such as:

``` text
Unknown patient ID
Missing required field
Unsupported tool
Executable action failure
```

are inserted into the next generation prompt.

The Designer is explicitly told to correct those problems without making
the environment trivial.

The loop therefore became:

``` text
Environment Designer
        |
        v
Candidate
        |
        v
Validator
        |
        +------ valid ------> Memory
        |
        +------ invalid
                 |
                 v
          Validation errors
                 |
                 +------------+
                              |
                              v
                     Environment Designer
                       next attempt
```

This was closer to the iterative environment-generation behavior I
wanted from the SPADE-inspired design.

## 22. Testing the Retry Mechanism Without Wasting API Calls

I did not want to wait for the LLM to randomly generate an invalid
environment just to see whether retry worked.

Instead, I wrote a controlled test using a fake Designer.

The first call deliberately returned:

``` python
{
    "id": "controlled_invalid_candidate"
}
```

The Validator correctly reported:

``` text
Missing required fields:
description
evaluation_criteria
ticket
```

The second fake generation call then received those exact errors
through:

``` text
validation_feedback
```

and returned a valid candidate.

The result was:

``` text
Generation attempt 1/3
Designer received feedback: None
Schema valid: False

Generation attempt 2/3
Designer received feedback:
[validation errors from attempt 1]

Schema valid: True

Final valid: True
```

This verified the complete feedback plumbing without making unnecessary
API calls.

## 23. Preventing Invalid Environments from Polluting Memory

While implementing retry, I noticed another problem.

If all three attempts failed, the previous version could still store the
final invalid task in Environment Memory.

That would be dangerous because later generations could use an invalid
environment as curriculum context.

I changed the loop so that only successfully validated environments
enter the curriculum memory.

The final rule is:

``` text
Generated
   |
   v
Validated?
   |
   +-- No --> retry / discard
   |
   +-- Yes
         |
         v
   Complexity score
         |
         v
       Memory
```

This keeps the curriculum grounded only in executable environments.

## 24. Iteration 5

After the feedback and retry pipeline was complete, I generated another
real environment.

Iteration 5 produced:

``` text
spade_clinical_pneumonia_sepsis_p001_v5
```

It passed on the first attempt:

``` text
Schema valid:       True
Executable valid:   True
```

Its complexity score was:

``` text
26.0
```

The final five-iteration experiment therefore produced:

  Iteration   Environment             Difficulty     Complexity
  ----------- ----------------------- ------------ ------------
  1           Pulmonary embolism      Moderate             17.5
  2           Submassive PE           Hard                 18.5
  3           DKA                     Hard                 18.5
  4           Urgent upper GI bleed   Hard                 23.5
  5           Pneumonia + sepsis      Hard                 26.0

The most useful part of this experiment was not simply that the final
number was higher.

The interesting behavior was:

``` text
17.5
  |
18.5
  |
18.5    <- curriculum plateau
  |
  |     explicit previous-complexity feedback added
  v
23.5
  |
26.0
```

Iteration 3 showed that an iteration counter alone did not guarantee
progressively harder environments.

After the previous validated complexity became explicit feedback, the
next environments became substantially more involved while continuing to
pass executable validation.

## 25. Final Implementation

The project-specific code is organized as:

``` text
spade_designer/
├── __init__.py
├── designer.py
├── validator.py
├── memory.py
├── run_loop.py
└── generated/
    ├── candidate_001.json
    ├── candidate_002.json
    ├── candidate_003.json
    ├── candidate_004.json
    ├── candidate_005.json
    └── memory.json
```

### `designer.py`

Responsible for:

``` text
corpus grounding
patient database grounding
previous-environment memory
complexity feedback
validation feedback
LLM generation
```

### `validator.py`

Responsible for:

``` text
schema validation
patient-reference validation
Healthcare_GYM execution
required-action testing
complexity measurement
```

### `memory.py`

Stores validated generated environments and their complexity
information.

### `run_loop.py`

Connects everything:

``` text
generate
-> validate
-> retry with feedback if needed
-> calculate complexity
-> store successful environment
-> continue curriculum
```

### `generated/`

Contains the actual outputs from the five experimental iterations.

I kept these files because they make the progression reproducible and
inspectable rather than reporting only summary numbers.

## 26. Running the Designer

First install Healthcare_GYM and its dependencies.

From the repository root:

``` bash
pip install -e .
```

Set an OpenAI API key in the environment:

``` bash
export OPENAI_API_KEY="..."
```

The Designer currently uses:

``` text
gpt-5-mini
```

To run the default loop:

``` bash
python -m spade_designer.run_loop
```

Or run a specific range manually:

``` bash
python - <<'PY'
from spade_designer.run_loop import run_spade_loop

run_spade_loop(
    start_iteration=6,
    num_iterations=1,
)
PY
```

Each iteration prints:

``` text
generation attempt
generated task ID
schema validation result
executable validation result
complexity score
memory status
```

Generated environments are saved under:

``` text
spade_designer/generated/
```

## 27. What Is SPADE and What Is My Adaptation?

I tried to keep this distinction explicit throughout the implementation.

The main ideas I took from SPADE were:

``` text
Environment Designer
corpus-grounded generation
complete multi-step environments
environment validation
accumulated environment memory
progressively challenging environments
feedback into later environment design
```

However, this project is **not an exact SPADE reproduction**.

Original SPADE uses an agent/policy and evaluates environments using
agent performance, including its hint-based regret signal.

This assignment specifically focuses on the Environment Designer and
does not require policy training.

Therefore, I replaced that part with:

``` text
structural complexity feedback
+
validation feedback
+
validated environment memory
```

Another important adaptation is the output format.

SPADE's tool-use Environment Designer can generate executable Python
environment implementations.

Healthcare_GYM already provides the executable runtime, tools, state,
and Gym interface.

Therefore, my Designer generates:

``` text
Healthcare_GYM-compatible task specifications
```

and then proves that those specifications are executable by injecting
them into the real Healthcare_GYM environment.

So the relationship is:

  SPADE concept            This project
  ------------------------ -----------------------------------
  Environment Designer     LLM task designer
  Corpus                   Existing Healthcare_GYM tasks
  Environment state        Healthcare_GYM `db.json`
  Generated environment    Healthcare_GYM task JSON
  Executable environment   `BioAgentGymEnv` + generated task
  Validation               Schema + real Gym execution
  Environment memory       `generated/memory.json`
  Designer feedback        Complexity + validation feedback
  Policy training          Intentionally not implemented

## 28. What I Learned From the Implementation

The most important part of this assignment was not calling an LLM to
generate JSON.

The difficult part was understanding **what an environment means inside
the library I selected**.

At the beginning, I thought I might need to generate entire Gym classes.

After studying Healthcare_GYM, I realized that the library already
separates task specification from runtime state and tools. That made
task generation a much more natural integration point.

The second major lesson was that structural validity is not enough.

A generated healthcare task can look reasonable while depending on
information that does not exist in the simulated world. This is why
database grounding and executable validation became central parts of the
implementation.

The third lesson came from the Iteration 3 plateau.

Simply telling an LLM to make each iteration harder did not guarantee a
meaningful curriculum. Giving the Designer explicit information about
previous validated environments and their complexity produced a much
clearer progression.

The final system therefore evolved from:

``` text
LLM -> JSON
```

into:

``` text
Corpus + World State + Memory
             |
             v
       LLM Designer
             |
             v
      Candidate Task
             |
             v
         Validator
        /         \
     valid       invalid
       |            |
       v            |
 Complexity         |
       |            |
       v            |
    Memory          |
       |            |
       +------> Designer
               feedback
```

That final loop is the main result of this assignment.
