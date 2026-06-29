import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ── 1. CONFIG ──────────────────────────────────────────────────────────────────

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = "./mistral-lora-ita"
ADAPTER_DIR = "./mistral-lora-adapter"


# ── 2. DATASET ─────────────────────────────────────────────────────────────────

data = [
    {"text": "### Domanda: Cos'è il machine learning?\n### Risposta: Il machine learning è una branca dell'intelligenza artificiale che consente ai sistemi di apprendere automaticamente dai dati senza essere esplicitamente programmati."},
    {"text": "### Domanda: Cos'è una rete neurale?\n### Risposta: Una rete neurale è un modello computazionale ispirato al cervello umano, composto da strati di neuroni artificiali interconnessi che elaborano informazioni."},
    {"text": "### Domanda: Cos'è il deep learning?\n### Risposta: Il deep learning è una sottocategoria del machine learning che utilizza reti neurali profonde con molteplici strati nascosti per apprendere rappresentazioni complesse dei dati."},
    {"text": "### Domanda: Cos'è il trasferimento di apprendimento?\n### Risposta: Il trasferimento di apprendimento è una tecnica che consente di applicare la conoscenza acquisita da un modello pre-addestrato a un nuovo problema correlato, riducendo tempi e dati necessari."},
    {"text": "### Domanda: Cos'è il reinforcement learning?\n### Risposta: Il reinforcement learning è un paradigma in cui un agente impara a prendere decisioni interagendo con un ambiente e ricevendo ricompense o penalità in base alle azioni compiute."},
    {"text": "### Domanda: Cos'è il natural language processing?\n### Risposta: Il natural language processing (NLP) è una branca dell'intelligenza artificiale che si occupa dell'interazione tra computer e linguaggio umano, permettendo alle macchine di comprendere, interpretare e generare testo in modo naturale."},
    {"text": "### Domanda: Cos'è un transformer?\n### Risposta: Un transformer è un'architettura di rete neurale basata sul meccanismo di self-attention, introdotta nel 2017 nel paper 'Attention Is All You Need'. È alla base di modelli come BERT, GPT e Mistral."},
    {"text": "### Domanda: Cos'è RAG nel contesto degli LLM?\n### Risposta: RAG (Retrieval-Augmented Generation) è una tecnica che combina un modello generativo con un sistema di recupero di documenti esterni, permettendo all'LLM di rispondere con informazioni aggiornate senza necessità di ri-addestramento."},
    {"text": "### Domanda: Cos'è un large language model?\n### Risposta: Un large language model (LLM) è un modello di deep learning addestrato su enormi quantità di testo per comprendere e generare linguaggio naturale. Esempi noti sono GPT-4, Mistral e LLaMA."},
    {"text": "### Domanda: Cos'è l'attenzione nei transformer?\n### Risposta: Il meccanismo di attenzione permette al modello di pesare l'importanza di ogni token in relazione agli altri durante l'elaborazione, consentendo di catturare dipendenze a lungo raggio nel testo."},
    {"text": "### Domanda: Cos'è il tokenization?\n### Risposta: La tokenization è il processo di suddivisione del testo in unità più piccole chiamate token, che possono essere parole, sotto-parole o caratteri. È il primo passo nell'elaborazione del testo per i modelli di linguaggio."},
    {"text": "### Domanda: Cos'è BERT?\n### Risposta: BERT (Bidirectional Encoder Representations from Transformers) è un modello di linguaggio di Google basato sui transformer che legge il testo in entrambe le direzioni, ed è particolarmente efficace per task di comprensione del linguaggio."},
    {"text": "### Domanda: Cos'è il fine-tuning?\n### Risposta: Il fine-tuning è il processo di ulteriore addestramento di un modello pre-addestrato su un dataset specifico per adattarlo a un compito particolare, mantenendo la conoscenza generale acquisita in precedenza."},
    {"text": "### Domanda: Cos'è LoRA?\n### Risposta: LoRA (Low-Rank Adaptation) è una tecnica di fine-tuning efficiente che aggiunge piccole matrici addestrabili a basso rango ai layer del modello, riducendo drasticamente il numero di parametri da aggiornare."},
    {"text": "### Domanda: Cos'è QLoRA?\n### Risposta: QLoRA combina la quantizzazione a 4-bit del modello base con la tecnica LoRA, permettendo il fine-tuning di modelli molto grandi su GPU consumer riducendo l'uso di memoria fino all'80% rispetto al fine-tuning classico."},
    {"text": "### Domanda: Cos'è la quantizzazione di un modello?\n### Risposta: La quantizzazione è una tecnica che riduce la precisione numerica dei pesi di un modello (ad esempio da 32-bit a 4-bit), riducendo memoria e velocizzando l'inferenza con una minima perdita di accuratezza."},
    {"text": "### Domanda: Cos'è PEFT?\n### Risposta: PEFT (Parameter-Efficient Fine-Tuning) è una famiglia di tecniche, tra cui LoRA e prompt tuning, che permette di adattare modelli pre-addestrati con un numero molto ridotto di parametri addestrabili, risparmiando risorse computazionali."},
    {"text": "### Domanda: Cos'è l'overfitting?\n### Risposta: L'overfitting si verifica quando un modello impara troppo bene i dati di training, incluso il rumore, perdendo la capacità di generalizzare su dati nuovi. Si contrasta con tecniche come il dropout, la regolarizzazione e l'aumento dei dati."},
    {"text": "### Domanda: Cos'è l'underfitting?\n### Risposta: L'underfitting si verifica quando un modello è troppo semplice per catturare i pattern nei dati, risultando in prestazioni scarse sia sul training set che sul test set."},
    {"text": "### Domanda: Cos'è la backpropagation?\n### Risposta: La backpropagation è l'algoritmo usato per addestrare le reti neurali, che calcola il gradiente della funzione di loss rispetto ai pesi propagando l'errore all'indietro attraverso i layer della rete."},
    {"text": "### Domanda: Cos'è il gradient descent?\n### Risposta: Il gradient descent è un algoritmo di ottimizzazione che aggiorna iterativamente i pesi di un modello nella direzione opposta al gradiente della funzione di loss, minimizzando progressivamente l'errore."},
    {"text": "### Domanda: Cos'è un iperparametro?\n### Risposta: Un iperparametro è un parametro del processo di training che non viene appreso dai dati ma impostato manualmente, come il learning rate, il numero di epoche o la dimensione del batch."},
    {"text": "### Domanda: Cos'è il dropout?\n### Risposta: Il dropout è una tecnica di regolarizzazione che durante il training disattiva casualmente una percentuale di neuroni a ogni iterazione, riducendo l'overfitting e migliorando la generalizzazione del modello."},
    {"text": "### Domanda: Cos'è una rete convoluzionale?\n### Risposta: Una rete neurale convoluzionale (CNN) è un'architettura specializzata per l'elaborazione di dati con struttura a griglia come le immagini, che usa filtri convoluzionali per estrarre automaticamente feature spaziali."},
    {"text": "### Domanda: Cos'è un autoencoder?\n### Risposta: Un autoencoder è una rete neurale che impara a comprimere i dati in una rappresentazione latente a bassa dimensionalità e poi a ricostruirli, usata per riduzione della dimensionalità e rilevamento di anomalie."},
    {"text": "### Domanda: Cos'è una GAN?\n### Risposta: Una GAN (Generative Adversarial Network) è composta da due reti in competizione: un generatore che crea dati sintetici e un discriminatore che li distingue dai reali. Questa dinamica porta il generatore a produrre dati sempre più realistici."},
    {"text": "### Domanda: Cos'è l'embedding?\n### Risposta: Un embedding è una rappresentazione vettoriale densa di dati discreti come parole o token in uno spazio continuo a bassa dimensionalità, dove elementi semanticamente simili sono vicini nello spazio vettoriale."},
    {"text": "### Domanda: Cos'è la normalizzazione dei dati?\n### Risposta: La normalizzazione dei dati è il processo di trasformazione delle feature in un intervallo comune, tipicamente [0,1] o con media zero e varianza unitaria, per migliorare la stabilità e la velocità del training."},
    {"text": "### Domanda: Cos'è la cross-entropy loss?\n### Risposta: La cross-entropy loss è una funzione di perdita usata nei problemi di classificazione che misura la differenza tra la distribuzione di probabilità predetta dal modello e quella reale delle etichette."},
    {"text": "### Domanda: Cos'è il learning rate?\n### Risposta: Il learning rate è un iperparametro che controlla la dimensione dei passi con cui il modello aggiorna i propri pesi durante il training. Un valore troppo alto causa instabilità, uno troppo basso rallenta la convergenza."},
]

dataset = Dataset.from_list(data)


# ── 3. MODEL LOADING (QLoRA — 4-bit) ──────────────────────────────────────────

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)


# ── 4. LoRA CONFIG ─────────────────────────────────────────────────────────────

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

model.print_trainable_parameters()
# Output atteso: trainable params ~2M su ~3.5B (~0.06%)


# ── 5. TRAINING ────────────────────────────────────────────────────────────────

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_strategy="epoch",
    max_length=256,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()


# ── 6. SAVE ADAPTER ────────────────────────────────────────────────────────────

model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Adapter salvato in: {ADAPTER_DIR}")
