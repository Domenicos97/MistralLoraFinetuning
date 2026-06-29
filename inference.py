import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


MODEL_NAME = "mistralai/Mistral-7B-v0.1"
ADAPTER_DIR = "./mistral-lora-adapter"


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    return model, tokenizer


def generate(model, tokenizer, question: str, max_new_tokens: int = 150) -> str:
    prompt = f"### Domanda: {question}\n### Risposta:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Restituisce solo la risposta generata
    return full_text.split("### Risposta:")[-1].strip()


if __name__ == "__main__":
    print("Caricamento modello...")
    model, tokenizer = load_model()

    questions = [
        "Cos'è il natural language processing?",
        "Cos'è un transformer?",
        "Cos'è RAG nel contesto degli LLM?",
    ]

    for q in questions:
        print(f"\nDomanda: {q}")
        print(f"Risposta: {generate(model, tokenizer, q)}")
        print("-" * 60)
