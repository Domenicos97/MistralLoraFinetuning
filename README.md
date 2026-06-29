# QLoRA Fine-tuning — Mistral 7B

Fine-tuning di **Mistral-7B-v0.1** con tecnica **QLoRA** (Quantized Low-Rank Adaptation) per risposta formale in italiano su domande di AI/ML.

## Cos'è QLoRA?

QLoRA combina due tecniche:
- **Quantizzazione 4-bit** del modello base → riduce drasticamente l'uso di memoria GPU
- **LoRA (Low-Rank Adaptation)** → aggiunge piccole matrici addestrabili ai layer del modello, lasciando i pesi originali congelati

Il risultato è un **adapter leggero** (~2M parametri addestrati su ~7B totali, circa 0.03%) che modifica il comportamento del modello senza riaddestrarlo interamente.

```
Modello base (congelato) + Adapter LoRA (addestrato) = Modello fine-tuned
```

## Stack Tecnologico

| Libreria | Ruolo |
|---|---|
| `transformers` | Caricamento modello e tokenizer |
| `peft` | Configurazione e gestione LoRA |
| `bitsandbytes` | Quantizzazione 4-bit |
| `trl` | Training con SFTTrainer |
| `datasets` | Gestione dataset |
| `accelerate` | Ottimizzazione hardware |

## Struttura del Progetto

```
MistralLoraFinetuning/
├── mistral_qlora_finetuning.ipynb  # Notebook Colab (consigliato)
├── finetune.py                      # Script di training standalone
├── inference.py                     # Script di inferenza
├── requirements.txt
└── README.md
```

## Quickstart — Google Colab

1. Apri `mistral_qlora_finetuning.ipynb` su Google Colab
2. Vai su **Runtime → Change runtime type → T4 GPU**
3. Esegui le celle in ordine

## Quickstart — Locale

> Richiede GPU con almeno 8GB VRAM e CUDA installato.

```bash
git clone https://github.com/Domenicos97/MistralLoraFinetuning
cd MistralLoraFinetuning
pip install -r requirements.txt

# Training
python finetune.py

# Inferenza
python inference.py
```

## Parametri LoRA

| Parametro | Valore | Descrizione |
|---|---|---|
| `r` | 8 | Rank delle matrici LoRA |
| `lora_alpha` | 16 | Scaling factor |
| `target_modules` | q_proj, v_proj | Layer da addestrare |
| `lora_dropout` | 0.05 | Dropout per regolarizzazione |

## Dataset

Il progetto include un dataset minimale di esempio (domande/risposte su AI in italiano) per dimostrare il formato atteso:

```
### Domanda: Cos'è il machine learning?
### Risposta: Il machine learning è una branca dell'intelligenza artificiale...
```

Per un fine-tuning reale, sostituire con un dataset più ampio nello stesso formato.

## Output

Dopo il training vengono prodotti:
- `./mistral-lora-ita/` — checkpoint del training
- `./mistral-lora-adapter/` — adapter LoRA finale (da usare per l'inferenza)

## Requisiti Hardware

| Ambiente | Compatibile |
|---|---|
| Google Colab (T4 gratuito) | ✅ |
| Google Colab Pro (A100) | ✅ |
| GPU locale ≥ 8GB VRAM | ✅ |
| CPU only | ❌ |

## Riferimenti

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Library](https://github.com/huggingface/peft)
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
