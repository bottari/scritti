# 🌒 **scritti**  
### *Applied LLM Experimentation, Evaluation, and Fine-Tuning*  

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![LLM R&D](https://img.shields.io/badge/Focus-LLM%20Evaluation%20%26%20Fine--Tuning-purple.svg)]()  
[![AI Governance](https://img.shields.io/badge/Discipline-AI%20Governance-orange.svg)]()

---

## 🧭 **Purpose & Professional Context**

This repository showcases my hands-on work in **LLM evaluation, dataset tooling, prompt orchestration, and small-scale fine-tuning** — the same domains I manage professionally as a **Technical Program Manager specializing in AI Operations & Systems Architecture**.

While my day-job focuses on *risk management, evaluation pipelines, and GenAI program execution*, this repo captures the **personal R&D work** I do to deepen the engineering/ML side of that discipline.

**In short:**  
> This is where I experiment with the models, metrics, and behaviors that I govern, evaluate, and operationalize in production environments.

---

## 🏛️ **Philosophy**

My philosophy is rooted in the punk-rock conviction that if the current infrastructure doesn't serve the vision, you build your own. I value the visceral over the performative. I don’t manage processes, I architect solutions that preserve the integrity of the original idea. I believe in the power of transparency in data, in art, and in character.

---

## 🧩 **What This Repository Demonstrates**

This repo is intentionally diverse — it contains a portfolio of practical, minimally-packaged scripts that demonstrate:

### 🛠️ Technical Engineering Capabilities
- **Model Comparison Frameworks:** Flask-based UIs for human-centric qualitative analysis.
- **Python Tooling:** Pandas for metric calculation, JSON parsing, and custom I/O utilities.
- **Custom Evaluation Metrics:** Work on iambic pentameter scoring and haiku verification.
- **Fine-Tuning:** Llama 3.1 8B and GPT-2 experiments with LoRA and partial layer unfreezing.
- **Quantization:** Optimization for local execution on limited hardware (16GB VRAM).

### 🧪 AI Governance & Model Lifecycle Relevance
- **HITL Pipelines:** Designing interfaces that bridge raw model output with human judgment.
- **Failure-Mode Analysis:** Identifying semantic drift, instability, and overfitting.
- **Logging Patterns:** Data structures that mirror enterprise-grade evaluation pipelines.
- **Experiment Reproducibility:** Maintaining prompt-to-output linkage for downstream analysis.

### 📈 Program Management & Systems Thinking
- Translating ambiguous goals into measurable experiments  
- Structuring AI workflows with clear inputs/outputs  
- Aligning fine-tuning work with model safety/evaluation concerns  

---

## ⚖️ **NEW: Evaluation Pipeline**

This evaluation pipeline compares a base model, a LoRA-finetuned variant, and an optional human poetry reference set. It focuses on raw creative language properties before any downstream post-processing into strict forms.

**Metrics (purpose):**
- `imagery_density`: Proportion of words that match a sensory or natural imagery lexicon. Higher means richer sensory language.
- `lexical_novelty`: Fraction of tokens not present in the training corpus. Higher suggests fresher vocabulary.
- `semantic_drift_from_base`: Cosine distance between base and LoRA outputs (embedding space). Higher means stronger stylistic divergence.
- `metaphor_density`: Frequency of syntactic patterns like "X is Y", "X becomes Y", "X like Y", "X of Y" using spaCy dependencies. Higher indicates more figurative structure.
- `surreal_imagery_score`: Mean semantic distance between adjacent nouns. Higher suggests more unusual juxtapositions.
- `surrealism_index`: Sum of `surreal_imagery_score` + `metaphor_density` + `imagery_density`. Higher means denser surreal language.
- `token_distribution_shift`: KL divergence between base and LoRA token frequency distributions. Higher means broader distribution shift.

**How to run:**
```bash
python C:\Users\micha\Desktop\projects\scritti\eval\run_eval.py --config C:\Users\micha\Desktop\projects\scritti\eval_config.yaml
```

**How to interpret the plots:**
- Imagery and novelty distributions: Look for shifts in median and tails. A rightward shift indicates richer imagery or higher novelty.
- Semantic drift scatter: Points farther right are LoRA samples that diverge most from their base counterparts.
- Histogram comparison: Overlap shows similarity; separation indicates stylistic change.
- Surrealism index box plot: Higher medians indicate stronger surreal composition across imagery, metaphor, and noun distance.
- Latent poetry space (UMAP): Clusters and separation show how model outputs occupy different semantic regions. If colored by imagery, brighter regions indicate denser sensory language.

**Example summary table (expected shape):**

| metric                     | base | lora | human | delta_lora_vs_base | delta_human_vs_base |
|---------------------------|------|------|-------|--------------------|---------------------|
| imagery_density           | 0.41 | 0.72 | 0.65  | +75%               | +59%                |
| lexical_novelty           | 0.34 | 0.57 | 0.29  | +68%               | -15%                |
| semantic_drift_from_base  | 0.22 | 0.49 | 0.00  | +122%              | n/a                 |
| metaphor_density          | 0.12 | 0.38 | 0.21  | +216%              | +75%                |
| surrealism_index          | 0.75 | 1.47 | 1.02  | +96%               | +36%                |


## ⚖️ New: Human Evaluation & Side-by-Side Comparison

The latest evolution of **scritti** moves beyond automated metrics into **Human-in-the-Loop (HITL) evaluation**. I have implemented a Flask-based review application to perform rigorous side-by-side (A/B) testing between different model architectures.

### **The Workflow**
1.  **Orchestration:** Run identical prompts through both a **Llama 3.1 (QLoRA)** and a **GPT-2 (Fine-tuned)** model.
2.  **Blind Review:** Use the Flask UI (`poem_review_app/`) to compare outputs without knowing which model produced which result.
3.  **Qualitative Scoring:** Models are rated on a 1-5 scale across three dimensions:
    * **Originality:** Does it avoid clichés and "LLM-isms"?
    * **Emotion:** Does the output evoke the intended mood?
    * **Imagery:** How vivid and specific is the sensory language?
4.  **Audit Trail:** Results are captured in structured JSON, preserving the link between Prompt ID, Model Version, and Reviewer Scores.

> **Why this matters for AI Governance:** This mimics production-grade RLHF and red-teaming workflows, creating a consistent A/B evaluation process that can be audited for model drift or stylistic alignment.

---

## 🔥 **Featured Work: GPT-2 Fine-Tuning (Before vs After)**

Below is a real output comparison between **GPT-2** and my **fine-tuned model** using the same input prompt and generation settings. The fine-tuned model was trained to capture a distinct poetic style, producing surreal and borderline nonsensical imagery rather than conventional narrative, demonstrating the effect of stylistic fine-tuning.

<details>
<summary><strong>🧠 BASE (GPT2-Large) MODEL OUTPUT 🧠</strong> (click to expand)</summary>
<br/>
fleece lid of eternity<br>
<br>
$11.00<br>
<br>
We're selling fleece lids of eternity - just in time for the holidays! We've worked hard to bring you this wonderful product and now it's available for purchase. It comes with a free lifetime warranty so you can be sure that your lids are well-maintained.<br>
<br>
A beautiful fleece cover made from 100% cotton. We use premium quality, polyester fabric to ensure a long life and superior durability. The cover is cut to fit snugly around your head, and is perfect for those who don't like sleeves or are more comfortable with a hood.<br>
<br>
The cover is sewn with two layers of fine, durable thread (100% nylon).<br>
<br>
It features a unique magnetic clasp that allows you to easily change your lids (which are sewn on separately) when needed.<br>
<br>
Our goal was to create a product that you could trust to last and will never break. We know our customers will be very happy with their purchase.<br>
<br>
You'll need an HTML5 capable browser to see this content. Play Replay with sound Play with<br>
<br>
sound 00:00 00:00<br>
<br>
All orders are shipped through USPS First Class Mail (usually takes about 2-3 weeks). All packages will come with tracking information.<br>
<br>
We have several options to customize your fleece cover. You can add a logo, or simply change any of the colors you'd like. Simply put the order number (
</details>

<details>
<summary><strong>🎭 FINETUNED (POESIA DI MICHELE BOTTARI) MODEL OUTPUT 🎭</strong> (click to expand)</summary>
<br>
fleece lid of eternity<br>
with infinite tiny wings<br>
<br>
found myself face to face<br>
with an old woman<br>
with a cart full<br>
of little kids<br>
and I couldn’t speak<br>
and she wouldn’t let me<br> 
in, and then she<br>
stopped talking and<br>
kissed my forehead, and<br> 
we just lay there<br>
looking at each other, and<br>
I felt bad for even thinking about it,<br>
but I did, and now<br>
it’s past midnight and<br> 
everyone’s asleep, and<br>
I don’t want to go back<br>
to the city, it’s not safe, and<br>
my mind keeps racing, trying to find<br> 
the exit, but I only make it farther and<br> 
further into the miasma.<br>
<br>
0305 hours<br>
<br>
facing the blackness<br>
with quivering palms<br>
<br>
and ankles bound<br>
with a redneck bitch<br> 
who won’t stop calling me Baby<br>
until I give in and<br>
move to Texas, where<br>
they let you<br>
kiss the women’s restroom walls<br>
as easily as you can walk<br>
through the streets.<br>
<br>
11:30 am<br>
<br>
making the bed<br>
with one shoulder tied<br>
as if I were going to break it<br>
<br>
sooner or later I will<br>
and then what will I have<br>
but a broken body, a scattered mind<br>
<br>
</details>

---

## 🔥 **Featured Work: Llama 3.1 8B Fine-Tuning (Before vs After)**

Below is a real output comparison between **Llama 3.1** and my **fine-tuned model** using the same input prompt (i.e., "walking") and generation settings. 

<details>
<summary><strong>🧠 BASE (Llama 3.1 8B) MODEL OUTPUT 🧠</strong> (click to expand)</summary>
<br>
In a boat on Lake Geneva, you can come across an island with its own village and chapel. The Isle of St. Peter is located in Switzerland near Montreux.<br>
The island has been privately owned since 1907 by the family of Jean-Jacques de Senarclens d’Aigremont who bought it from Prince Albert I. In order to preserve this natural site as well as possible, only a few people are allowed on the small area during daytime hours: the
<br>
</details>

<details>
<summary><strong>🎭 FINETUNED (POESIA DI MICHELE BOTTARI) MODEL OUTPUT 🎭</strong> (click to expand)</summary>
<br>
trembling drenched white<br>
with fever<br>
and/or confusion<br>
<br>
you tremble and hide<br>
the shoes you've worn dancing<br>
<br>
*<br>
<br>
The original sin of this world<br>
is its inability to include<br>
all alliterations,<br>
and so I must alter it.<br>
<br>
*<br>
<br>
Her face in repose looks like a piano.<br>
<br>
We drank too much wine for this to be happening, I thought.<br>
I hadn't told her my feelings then, and now we're drinking again,<br>
and maybe I'll<br>
<br>
</details>

---

## 🔥 **Haiku generation results**

Below is an example of output from the haiku generation script.

<details>
<summary><strong>🎭 FINETUNED (POESIA DI MICHELE BOTTARI) MODEL OUTPUT 🎭</strong>(click to expand)</summary>
<br>

HAIKU GENERATOR - Sample First, Assemble ALL Possible (with spaCy)

Prompt: " <br>
fleece lid<br>
"<br>

Attempt #4

[1] Generating poetic text...

Generated text (full):<br>
fleece lid wristband crinkled  open to the light,  fingertips tracing smooth blue strokes  through tangled leaf tissue  gently pulling apart  a reddish featherfall flecked red against  blue.  Leaves swaying unsteady  against a summer wind,  taken from their source, deep in  the woods of Vermont.

[2] Extracting phrases from generated text (using spaCy)...

Found 21 phrases:
  1. (6 syl) fleece lid wristband crinkled
  2. (5 syl) fingertips pulling
  3. (11 syl) fingertips pulling a reddish featherfall
  4. (4 syl) fleece lid wristband
  5. (2 syl) the light
  6. (3 syl) smooth blue strokes
  7. (5 syl) tangled leaf tissue
  8. (6 syl) a reddish featherfall
  9. (4 syl) a summer wind
  10. (2 syl) their source
  11. (2 syl) the woods
  12. (3 syl) to the light
  13. (6 syl) through tangled leaf tissue
  14. (3 syl) against blue
  15. (6 syl) against a summer wind
  16. (3 syl) from their source
  17. (3 syl) in the woods
  18. (3 syl) of Vermont
  19. (12 syl) Leaves swaying unsteady  against a summer wind
  20. (5 syl) taken from their source
  ... and 1 more

[3] Searching for ALL 5-7-5 combinations...

✓ Found 6 valid haiku(s)!

🌿 ALL GENERATED HAIKUS:

Haiku #1 (5-7-5):<br>
  fingertips pulling<br>
  deep in  the woods of Vermont<br>
  tangled leaf tissue<br>

Haiku #2 (5-7-5):<br>
  fingertips pulling<br>
  deep in  the woods of Vermont<br>
  taken from their source<br>

Haiku #3 (5-7-5):<br>
  tangled leaf tissue<br>
  deep in  the woods of Vermont<br>
  fingertips pulling<br>

Haiku #4 (5-7-5):<br>
  tangled leaf tissue<br>
  deep in  the woods of Vermont<br>
  taken from their source<br>

Haiku #5 (5-7-5):<br>
  taken from their source<br>
  deep in  the woods of Vermont<br>
  fingertips pulling<br>

Haiku #6 (5-7-5):<br>
  taken from their source<br>
  deep in  the woods of Vermont<br>
  tangled leaf tissue<br>
<br>
</details>

---

## 🔥 **Iambic pentameter couplet generation results**

Below is an example of output from the iambic pentameter couplet script. Note that ***trochaic substitution*** is allowed.

<details>
<summary><strong>🎭 FINETUNED (POESIA DI MICHELE BOTTARI) MODEL OUTPUT 🎭</strong>(click to expand)</summary>
<br>
[Fine-tuned] ATTEMPT 5/25

[Fine-tuned] Generating 400 tokens...<br>
[Fine-tuned] Generated text:<br>
, not seeing anything. there's nothing left but a dim light on the back of her head and an air chill in her lungs as she closes eyes to contemplate the pain of it all, trying desperately for clarity. i couldn't stay still if i wanted some sleep, let alone those nights ahead…and when i woke up around 8:30am with my body shaking by the sudden urge from inside me that was too much like death and I had no place else than home. we were going out together….then something happened next door/door again. “well, then why didn’t you just open the window? because everyone is scared they'll get attacked or killed....”“it‘s true. so many people are afraid to come into their homes during quiet hours such how do anyone know who can help them escape while also getting shot down! [she doesn´re sure.]but...you're gonna leave your phone behind?"‌he nods hhopefully thinking about what he said before his friend picked him off-guard looking drunk maybe remembering this guy would have seen someone coming look after us until midnight once more..there‗s always been these moments where both parties  try harder to make sense; sometimes one forgets things even though everything goes better between them anyways.…somehow suddenly remembering another person could be confusing."did you think somebody might sneak over here under lights ?so fast~doppelgängrichterweimar really want security cameras???this guy isn¬ve ready!]․a little voice calls through our windowws wondering aloud whether whatever wasno making contact with anybody should arrive sooner ratherthan later anyway, which leaves no room anymore for any attempts at concealment.[1]definitely possible since sintra hasn—tted far enough above sea level thus now(to avoid hitting shore).— ”that sounds familiar. hes standing close beside lissey (meantfor reference... 


[Analysis] Extracting coherent phrases from generated text...<br>
[Analysis] Found 137 grammatical phrases<br>
[Analysis] Phrase breakdown: {'sentence': 14, 'clause': 7, 'noun_phrase': 77, 'verb_phrase': 39}<br>

[Analysis] Adjusting phrases to fit iambic pentameter...<br>
[Analysis] Found 0 phrases that fit IP through adjustment<br>

[Analysis] Looking for combinable phrases...<br>
[Analysis] Found 192 valid combinations<br>
[Analysis] Filtered out 0 lines with repeated words<br>
[Analysis] Total candidate IP lines: 192<br>
[Fine-tuned] Found 192 new candidate lines<br>

[Fine-tuned] Top 5 candidates from this attempt:
  1. [100] nothing clarity also getting shot
  2. [100] nothing the sudden urge happened again
  3. [100] nothing the window also getting shot
  4. [100] nothing another person closes eyes
  5. [100] nothing another person hitting shore

[Fine-tuned] Total accumulated candidates: 291

[Fine-tuned] Searching for rhyming couplet in 291 total candidates...<br>
[Fine-tuned] Requiring grammatical score >= 80/100<br>
[Fine-tuned] Candidates meeting threshold: 164/291<br>
[Pairing] Checking 164 quality candidates...<br>
[Pairing] ✓ Found couplet with score 91.5<br>

[Fine-tuned] ✓✓✓ COUPLET FOUND! ✓✓✓


FINAL RESULTS


Prompt: 'Coming upon the lake at night'

COHERENT IAMBIC PENTAMETER COUPLET:

***her dainty moorings trying anymore***<br>
***clarity our windows hitting shore***<br>

Attempts needed: 5<br>
Lines rhyme: True<br>
Semantic similarity: 0.576<br>
Word repetition (content): 0.0%<br>
N-gram overlap: {2: 0.0, 3: 0.0, 4: 0.0}<br>
Overall score: 91.5/100<br>

Line 1 Analysis:<br>
  Text: her dainty moorings trying anymore<br>
  End word: 'anymore'<br>
  Type: combined<br>
  Modification: none<br>
  Grammatical score: 100/100<br>
  Stress pattern: 0101010101<br>

Line 2 Analysis:<br>
  Text: clarity our windows hitting shore<br>
  End word: 'shore'<br>
  Type: combined<br>
  Modification: none<br>
  Grammatical score: 100/100<br>
  Stress pattern: 1001010101<br>
  ⚠ Trochaic substitution in foot/feet: [1]<br>

Original generated text:

, not seeing anything. there's nothing left but a dim light on the back of her head and an air chill in her lungs as she closes eyes to contemplate the pain of it all, trying desperately for clarity. i couldn't stay still if i wanted some sleep, let alone those nights ahead…and when i woke up around 8:30am with my body shaking by the sudden urge from inside me that was too much like death and I had no place else than home. we were going out together….then something happened next door/door again. “well, then why didn’t you just open the window? because everyone is scared they'll get attacked or killed....”“it‘s true. so many people are afraid to come into their homes during quiet hours such how do anyone know who can help them escape while also getting shot down! [she doesn´re sure.]but...you're gonna leave your phone behind?"‌he nods hhopefully thinking about what he said before his friend picked him off-guard looking drunk maybe remembering this guy would have seen someone coming look after us until midnight once more..there‗s always been these moments where both parties  try harder to make sense; sometimes one forgets things even though everything goes better between them anyways.…somehow suddenly remembering another person could be confusing."did you think somebody might sneak over here under lights ?so fast~doppelgängrichterweimar really want security cameras???[this guy isn¬ve ready!]․a little voice calls through our windowws wondering aloud whether whatever wasno making contact with anybody should arrive sooner ratherthan later anyway, which leaves no room anymore for any attempts at concealment.[1]definitely possible since sintra hasn—tted far enough above sea level thus now(to avoid hitting shore).— ”that sounds familiar. hes standing close beside lissey (meantfor reference    

Total candidates found: 291

COHERENCE METHOD SUMMARY

✓ Preserves actual sentences and phrases from generated text<br>
✓ Uses SpaCy to identify grammatical structures<br>
✓ Adjusts phrases minimally to fit iambic pentameter<br>
✓ Combines shorter phrases when needed<br>
✓ Scores lines by grammaticality and semantic coherence<br>
✓ Only accepts lines with high grammatical scores (≥80/100)<br>
✓ Rejects pairs with identical end words<br>
✓ Allows trochaic substitution (up to 2 feet)<br>
✓ Much more coherent than word-pool reconstruction!<br>
<br>
</details>

---

## 🤖 **Agent Mercury — Local Speech-Enabled LLM Agent**

This section contains the scaffolding for: **Agent Mercury**, a fully local, speech-enabled LLM agent built on my merged GPT-2 fine-tuned poetry model.

This agent demonstrates how to combine **local model inference**, **offline STT & TTS**, **memory-bounded conversational context**, and **agent-style prompting** into a cohesive interactive tool. It also reflects real model-ops considerations like device-aware loading, error handling, and multimodal input.

---

## 📁 Directory: `gpt2-files/local-agent-001/`

```text
local-agent-001/
├── main.py                          # Full agent logic (voice + text output & voice + text input)
├── main-voice-input-only.py         # Text output & voice + text input mode
├── requirements.txt                 # Dependencies (transformers, vosk, sounddevice, torch, etc.)
├── test_cuda.py                     # Quick GPU/torch diagnostic
├── agent-mercury/                   # Virtual environment (excluded via .gitignore)
├── models/
│   └── full_merged_gpt2-finetuned-poetry-mercury-04--copy-attempt/    # For example, not included in public repo 
│       ├── config.json
│       ├── generation_config.json
│       ├── merges.txt
│       ├── model.safetensors
│       └── vocab.json               
└── vosk-model-en-us-0.22/           # Offline STT model (excluded via .gitignore)
```

---

## 🧠 Agent Mercury Overview

Mercury is powered by the **fine-tuned GPT-2 poetry model** from this repo, combined with:

- **Offline speech recognition** using Vosk  
- **Dynamic conversation memory** via bounded `deque`  
- **Personality-driven system prompt** (“ancient poet and insightful assistant”)  
- **Ghost-word cleanup** for noisy speech transcripts  
- **Device-aware model loading** (GPU if available, CPU fallback)  
- **Robust generation loop** with sampling + repetition penalties  
- **Command interface** for managing agent state

This brings together multiple areas of experimentation from the repo — generation, fine-tuning, evaluation logic, and applied inference.

---

## 🧩 Example Functionality

- Listens for offline speech, cleans filler words, and converts to text  
- Builds a prompt using a sliding window of conversation memory  
- Generates responses using the fine-tuned GPT-2 model  
- Removes prompt leakage and ensures clean assistant output  
- Falls back to typed input if the microphone is silent  
- Warns and adjusts automatically if CUDA is unavailable

---

### **`First_Edition_GenPs-001_10_14_25.txt`** — Results examples
Examples of results from using these scripts for fine-tuning and generation experiments.

### **`/gpt2-files/`** — GPT-2 experiments
- **`generation/`**: Scripts for generating poetry in various forms (iambic pentameter, haiku) and interactive chat interfaces with model comparison
- **`tuning/`**: Fine-tuning scripts with partial layer unfreezing and line break preservation

### **`/llama-files/`** — Llama 3.1 8B experiments  
- **`generation/`**: Interactive poetry chat terminal for Llama 3.1 with comparison functionality
- **`tuning/`**: Fine-tuning scripts optimized for Llama 3.1 with 4-bit and 8-bit quantization

---

## 🧠 **Why This Repo Exists**

> I created **scritti** to make transparent that — despite building my career as a Technical Program Manager — I am also deeply hands-on with the *engineering and experimental* side of AI.

Where my résumé emphasizes:
- **risk mitigation**  
- **requirements definition**  
- **AI safety operations**  
- **cross-functional program delivery**

…this repo highlights the *technical underpinnings* I continue to develop:
- Model behavior analysis  
- Evaluation design  
- Scripting tooling  
- Experimentation discipline  

Together, they illustrate how I bridge **AI governance ↔ applied engineering**.

---

## 🧰 **Key Technologies Used**

- Codex
- Python (tooling, parsing, orchestration)
- PyTorch
- Transformers (Hugging Face)
- spaCy
- TensorBoard
- LoRA adapters
- Quantization / 8-bit loading
- Dataset cleaning and text normalization
- Prompt engineering & stylistic constraints

---

## 📚 **How This Aligns With My Resume**

| Resume Theme                 | What This Repo Shows                                            |
| ---------------------------- | --------------------------------------------------------------- |
| AI Governance                | Behavior analysis, evaluation logic, failure-mode understanding |
| Technical Program Management | Structured experiments, reproducibility, workflow design        |
| Python Tooling               | Scripts for formatting, parsing, evaluation                     |
| Model Evaluation             | Direct comparison harnesses + qualitative assessments           |
| LLM Training Knowledge       | Fine-tuning code, quantization, training loops                  |
| Cross-Functional Bridging    | Clear documentation, transparency, reproducible outputs         |

---

## 👤 **About Me**

**Michael Bottari**  
Technical Program Manager — AI Operations & Systems Architecture  
📧 [mfbottari@gmail.com](mailto:mfbottari@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/michael-bottari)  
💻 [GitHub](https://github.com/bottari)  
📺 [YouTube / Trochee Lab](https://www.youtube.com/@trochee_lab)

---

## 📝 **License**

MIT License — feel free to fork, study, and experiment.

