import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, PreTrainedModel


def attach_lora(model: PreTrainedModel) -> PeftModel:
    """Attach LoRA adapters to the Qwen policy."""
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def main() -> None:
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cpu")
    policy = attach_lora(base_model)
    policy.print_trainable_parameters()

    trainable_names = [
        name for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable_names, "No trainable parameters found."

    unexpected = [
        name for name in trainable_names
        if ".lora_A." not in name and ".lora_B." not in name
    ]
    assert not unexpected, f"Unexpected trainable parameters: {unexpected[:10]}"

    print(f"Example trainable parameter: {trainable_names[0]}")
    print("Local check passed: only LoRA adapter parameters are trainable.")


if __name__ == "__main__":
    main()