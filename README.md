# 🌒 **scritti**  
### *Applied LLM Experimentation, Evaluation, and Fine-Tuning*  

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![LLM R&D](https://img.shields.io/badge/Focus-LLM%20Evaluation%20%26%20Fine--Tuning-purple.svg)]()  
[![AI Governance](https://img.shields.io/badge/Discipline-AI%20Governance-orange.svg)]()

---

## 🧭 **Purpose & Professional Context**

This repository showcases my hands-on work in **LLM evaluation, dataset tooling, prompt orchestration, and small-scale fine-tuning** — the same domains I manage professionally as a **Technical Program Manager specializing in AI Governance & Applied AI Operations**.

While my day-job focuses on *risk management, evaluation pipelines, and GenAI program execution*, this repo captures the **personal R&D work** I do to deepen the engineering/ML side of that discipline.

**In short:**  
> This is where I experiment with the models, metrics, and behaviors that I govern, evaluate, and operationalize in production environments.

---

## 🧩 **What This Repository Demonstrates**

This repo is intentionally diverse — it contains a portfolio of practical, minimally-packaged scripts that demonstrate:

### 🛠️ Technical Engineering Capabilities
- Python tooling (Pandas, tokenization, JSON parsing, I/O utilities)  
- Prompt orchestration logic  
- Custom evaluation metrics (e.g., work on iambic pentameter scoring and haiku creation)  
- GPT-2 fine-tuning experiments with partial layer unfreezing  
- Quantization and optimization on limited hardware (16GB VRAM)  
- Dataset formatting utilities (for supervised fine-tuning or RAG prep)

### 🧪 AI Governance & Model Lifecycle Relevance
- Understanding of *why* models behave incoherently  
- Ability to perform controlled prompt-based evaluations  
- Failure-mode analysis (semantic drift, instability, overfitting)  
- Logging patterns that mirror internal evaluation pipelines  
- Experiment reproducibility practices  

### 📈 Program Management & Systems Thinking
- Translating ambiguous goals into measurable experiments  
- Structuring AI workflows with clear inputs/outputs  
- Aligning fine-tuning work with model safety/evaluation concerns  

---

## 🔥 **Featured Work: GPT-2 Fine-Tuning (Before vs After)**

Below is a real output comparison between **GPT-2** and my **fine-tuned model** using the same input prompt and generation settings. The fine-tuned model was trained to capture a distinct poetic style, producing surreal and borderline nonsensical imagery rather than conventional narrative, demonstrating the effect of stylistic fine-tuning.

<details>
<summary><strong>🧠 BASE (GPT2) MODEL OUTPUT 🧠</strong> (click to expand)</summary>
<br/>
watching with glacial gravity isn't life. She knew he wasn't gay. What they didn't know, until her the hospital stopped billing him for $75 a week—two or three quarters of what he paid in back fees—was that she was pushing him toward that home over there where the men let more money run through the plumbing and had greater bandwidth for the conversation, by contrast, if someone had only run an electric line down in time and slid money back into himself through life.

"She was someone of stature and quality, still," Drew Alexander continues. "[He] wanted someone nice. Nice how people say, if it was to just tell the next one about, I don't think the price would ever<br>
 work because you never tell anybody that you work in white tape after 'hand money,' I'm happy to work in handshake money for, ya know, sorta like their life... He could be spooked by all the pushy parties. Or who would believe this is a coincidence, who would allow himself to be cuckolded, his wife<br>
 cheated on by the roommate… they always blamed [The Purge writer] Jackson, nothing could have survived outside the U.K.s conditions like that."[10] But according to Andrew Maguire (TAS and prolific Purge eponym), Alexander's willingness to play with facts wasn't seen at the 2007 Indianapolis premiere for his 2009 instalment Left Me Stand Alone when authorities declared white trash vans roaming the heart of town were highly controversial responses to drug rapes. "Some evil little drug cunts like
 Margo Irving should be responsible for such barbaric behavior because she let her child be hit by a<br>
 semi-maimed alcoholic male … they fought every step of the way like this when under the influence that proves how cheap were they and still are." Every one. [F/L/G is valid opinions]. As women the environment rarely deterr the vigilante elements.[Orness suggests the oldman played you falsely... as always, heart][Image so wonderfully bare one where gore is spilled.[Male depiction?]Now wait a minute- Apparently just leaving this low bitch tag intact[Still "nice how you know it!" show] http://bit.ly/8VA91U . Ever spend all week this Newer standard bill containing $1 MillionFPS HOF win awards the<br>
 boy whom to love was connected: articleIt also said a fuck even more popular.Time: How Lit. 19 r i D ************* *****U t/H dade.ruskus,last sings on so donner 35S ALLR ins good word well g l iv do<br>
 r all on imafl,we the skingo.alevich Suck better price milano butget like assemede sonodaisnnot me second free phone thisc:1k800 an infinity 13 j u let frather' 11D freesia play the negard Stakedhell<br>
 But three popindteddu-four fine answers ifeven commacedte1ibands,"$37636 O.O baic ... D any chance you can make an honorable answer?5 out963 plus number".Keep records it is important sish sayitsprayshop checks false abuse."They became … James Row and so deseewedhip he marm a call had could too vei-<br>
 alsostoterseaktion nehow much that night on bodymapaysha servicethey chekepsed hot stay feel os back these G iv get mad that butse inside my yors. Seen my pro CASS WebletIt was down still which rhou d eherec with nuk

</details>

<details>
<summary><strong>🎭 FINETUNED (POESIA DI MICHELE BOTTARI) MODEL OUTPUT 🎭</strong> (click to expand)</summary>
</br>
watching with glacial intensity <br>
pretending not to notice, like you can do it <br>
without thoughts, without trying, <br>
like two trapped souls in searing and twisting <br>
lens shadows. <br>
<br>
<br>
I'm sorry, I don't know what else to say. <br>
        I never knew that you could be so loving, <br>
even when things are hard, when the sun <br>
and the empty streets hang motionless <br>
in a pall of gray, and <br>
the mourners in white sit on gray <br>
trees writhing in the cool air <br>
but I now understand <br>
what it means to feel the depth of <br>
a loved one's sorrow, to understand <br>
how one's love can survive <br>
the ravages of time, the rigors of <br>
deserts and fires and tornados, <br>
and how we can stop believing <br>
when everything else crumbles, and <br>

</details>

---

## 🔥 **Haiku generation results**

Below is an example of output from the haiku generation script.

<details>
<summary><strong>🎭 FINETUNED (POESIA DI MICHELE BOTTARI) MODEL OUTPUT 🎭</strong>(click to expand)</summary>
<br>

HAIKU GENERATOR - Sample First, Assemble ALL Possible (with spaCy)

Prompt: " 
fleece lid of eternity
near silvery dunes
"

Attempt #5

[1] Generating poetic text...

Generated text (full):
fleece lid of eternity near silvery dunes I watched as you made the path taking in everything from end to end you even paused for a moment at the bottom of that sand dune, almost looking up didn't think anyone would look at you like that but no one did and now we are here together again, gathering dust

[2] Extracting phrases from generated text (using spaCy)...

Found 23 phrases:
  1. (2 syl) I watched
  2. (2 syl) you made
  3. (4 syl) the path taking
  4. (7 syl) the path taking everything
  5. (2 syl) you paused
  6. (5 syl) anyone would look
  7. (3 syl) no one did
  8. (4 syl) silvery dunes
  9. (2 syl) the path
  10. (3 syl) a moment
  11. (3 syl) the bottom
  12. (3 syl) that sand dune
  13. (2 syl) no one
  14. (5 syl) of eternity
  15. (5 syl) near silvery dunes
  16. (2 syl) from end
  17. (2 syl) to end
  18. (4 syl) for a moment
  19. (4 syl) at the bottom
  20. (4 syl) of that sand dune
  ... and 3 more

[3] Searching for ALL 5-7-5 combinations...

✓ Found 6 valid haiku(s)!

🌿 ALL GENERATED HAIKUS:

Haiku #1 (5-7-5):
  anyone would look<br>
  the path taking everything<br>
  of eternity<br>

Haiku #2 (5-7-5):
  anyone would look<br>
  the path taking everything<br>
  near silvery dunes<br>

Haiku #3 (5-7-5):
  of eternity<br>
  the path taking everything<br>
  anyone would look<br>

Haiku #4 (5-7-5):
  of eternity<br>
  the path taking everything<br>
  near silvery dunes<br>

Haiku #5 (5-7-5):
  near silvery dunes<br>
  the path taking everything<br>
  anyone would look<br>

Haiku #6 (5-7-5):
  near silvery dunes<br>
  the path taking everything<br>
  of eternity<br>

🌿 SUMMARY: Found 8 total haiku(s) across all attempts

Showing 8 unique haiku(s):

#1:
  of eternity<br>
  a momentary reprieve<br>
  near silvery dunes<br>

#2:
  near silvery dunes<br>
  a momentary reprieve<br>
  of eternity<br>

#3:
  anyone would look<br>
  the path taking everything<br>
  of eternity<br>

#4:
  anyone would look<br>
  the path taking everything<br>
  near silvery dunes<br>

#5:
  of eternity<br>
  the path taking everything<br>
  anyone would look<br>

#6:
  of eternity<br>
  the path taking everything<br>
  near silvery dunes<br>

#7:
  near silvery dunes<br>
  the path taking everything<br>
  anyone would look<br>

#8:
  near silvery dunes<br>
  the path taking everything<br>
  of eternity<br>

✓ Saved 8 unique haikus to: haiku-output-20251122_201751.txt
<br>
</details>

---

## 🔥 **Iambic pentameter couplet generation results**

Below is an example of output from the iambic pentameter couplet script. Note that ***trochaic substitution*** is allowed.

<details>
<summary><strong>🎭 FINETUNED (POESIA DI MICHELE BOTTARI) MODEL OUTPUT 🎭</strong>(click to expand)</summary>
<br>
[Fine-tuned] ATTEMPT 5/25

[Fine-tuned] Generating 400 tokens...
[Fine-tuned] Generated text:
, not seeing anything. there's nothing left but a dim light on the back of her head and an air chill in her lungs as she closes eyes to contemplate the pain of it all, trying desperately for clarity. i couldn't stay still if i wanted some sleep, let alone those nights ahead…and when i woke up around 8:30am with my body shaking by the sudden urge from inside me that was too much like death and I had no place else than home. we were going out together….then something happened next door/door again. “well, then why didn’t you just open the window? because everyone is scared they'll get attacked or killed....”“it‘s true. so many people are afraid to come into their homes during quiet hours such how do anyone know who can help them escape while also getting shot down! [she doesn´re sure.]but...you're gonna leave your phone behind?"‌he nods hhopefully thinking about what he said before his friend picked him off-guard looking drunk maybe remembering this guy would have seen someone coming look after us until midnight once more..‏there‗s always been these moments where both parties  try harder to make sense; sometimes one forgets things even though everything goes better between them anyways.…somehow suddenly remembering another person could be confusing."did you think somebody might sneak over here under lights ?so fast~doppelgängrichterweimar really want security cameras???[‮this guy isn¬ve ready!]․a little voice calls through our windowws wondering aloud whether whatever wasno making contact with anybody should arrive sooner ratherthan later anyway, which leaves no room anymore for any attempts at concealment.[1]definitely possible since sintra hasn—tted far enough above sea level thus now(to avoid hitting shore).— ”that sounds familiar. hes standing close beside lissey (meantfor reference... 


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

, not seeing anything. there's nothing left but a dim light on the back of her head and an air chill in her lungs as she closes eyes to contemplate the pain of it all, trying desperately for clarity. i couldn't stay still if i wanted some sleep, let alone those nights ahead…and when i woke up around 8:30am with my body shaking by the sudden urge from inside me that was too much like death and I had no place else than home. we were going out together….then something happened next door/door again. “well, then why didn’t you just open the window? because everyone is scared they'll get attacked or killed....”“it‘s true. so many people are afraid to come into their homes during quiet hours such how do anyone know who can help them escape while also getting shot down! [she doesn´re sure.]but...you're gonna leave your phone behind?"‌he nods hhopefully thinking about what he said before his friend picked him off-guard looking drunk maybe remembering this guy would have seen someone coming look after us until midnight once more..‏there‗s always been these moments where both parties  try harder to make sense; sometimes one forgets things even though everything goes better between them anyways.…somehow suddenly remembering another person could be confusing."did you think somebody might sneak over here under lights ?so fast~doppelgängrichterweimar really want security cameras???[‮this guy isn¬ve ready!]․a little voice calls through our windowws wondering aloud whether whatever wasno making contact with anybody should arrive sooner ratherthan later anyway, which leaves no room anymore for any attempts at concealment.[1]definitely possible since sintra hasn—tted far enough above sea level thus now(to avoid hitting shore).— ”that sounds familiar. hes standing close beside lissey (meantfor reference    

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

## 🤖 **New: Agent Mercury — Local Speech-Enabled LLM Agent**

A new addition to the repository: **Mercury**, a fully local, speech-enabled LLM agent built on my merged GPT-2 fine-tuned poetry model.

This agent demonstrates how to combine **local model inference**, **offline STT**, **memory-bounded conversational context**, and **agent-style prompting** into a cohesive interactive tool. It also reflects real model-ops considerations like device-aware loading, error handling, and multimodal input.

---

## 📁 Directory: `gpt2-files/local-agent-001/`

```text
local-agent-001/
├── main.py                          # Full agent (text + optional voice input)
├── main-voice-input-only.py         # Voice-only simplified entry point
├── requirements.txt                 # Dependencies (transformers, vosk, sounddevice, torch, etc.)
├── test_cuda.py                     # Quick GPU/torch diagnostic

├── agent-mercury/                   # Agent logic, utilities, prompt handling
│   └── ...                          # (Modularized helper scripts)

├── models/
│   └── full_merged_gpt2-finetuned-poetry-mercury-04--copy-attempt/
│       ├── config.json
│       ├── generation_config.json
│       ├── merges.txt
│       ├── model.safetensors
│       └── vocab.json               # Fine-tuned + merged GPT-2 weights

└── vosk-model-en-us-0.22/           # Offline speech-recognition model for voice mode
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

## ▶️ Running the Agent

### **Text + Voice Mode (full agent)** - (Note: this one is under construction)
```bash
python main.py
```

### **Voice-Only Mode** - (Note: this one is the most operational currently)
```bash
python main-voice-input-only.py
```

---

## 🎤 Voice Commands

When running the full agent (`main.py`):

- `/voice` — Toggle voice input (requires Vosk model loaded)  
- `/clear` — Reset conversation memory  
- `/exit` — Quit cleanly  
- `/quit` or `/bye` — Same as above  

---

## 🧩 Example Functionality

- Listens for offline speech, cleans filler words, and converts to text  
- Builds a prompt using a sliding window of conversation memory  
- Generates responses using the fine-tuned GPT-2 model  
- Removes prompt leakage and ensures clean assistant output  
- Falls back to typed input if the microphone is silent  
- Warns and adjusts automatically if CUDA is unavailable

---

## 📁 **Repository Structure**

```text
📁 scritti/
├── 📄 README.md
├── 📄 First_Edition_GenPs-001_10_14_25.txt

├── 📁 gpt2-files/
│   ├── 📁 generation/
│   │   ├── 📄 gpt2-generate-iambic-pentameter-006--couplets-spacy.py
│   │   ├── 📄 gpt2-generation-haiku_form-004-smaller-phrases--as-many-as-possible.py
│   │   └── 📄 interactive-poetry-chat-in-terminal-002--gpt2-with-comparison.py
│   │
│   ├── 📁 tuning/
│   │   └── 📄 the-gpt2-fine-tuning-script-thats-the-best-tweaked-003-unfreeze-top-layers---keep-source-line-breaks.py
│   │
│   └── 📁 local-agent-001/
│       ├── 📄 main-voice-input-only.py
│       ├── 📄 main.py
│       ├── 📄 requirements.txt
│       ├── 📄 test_cuda.py
│       │
│       ├── 📁 agent-mercury/
│       │   └── ... (agent logic, prompt handling, utilities)
│       │
│       ├── 📁 models/
│       │   └── 📁 full_merged_gpt2-finetuned-poetry-mercury-04--copy-attempt/
│       │       ├── 📄 config.json
│       │       ├── 📄 generation_config.json
│       │       ├── 📄 merges.txt
│       │       ├── 📄 model.safetensors
│       │       └── 📄 vocab.json
│       │
│       └── 📁 vosk-model-en-us-0.22/
│           └── ... (offline STT model files)

└── 📁 llama-files/
    ├── 📁 generation/
    │   └── 📄 interactive-poetry-chat-in-terminal-for-llama3-002-with-comparison.py
    └── 📁 tuning/
        └── 📄 fine-tuning-script-for-llama-3-q4-001.py
```

### **`First_Edition_GenPs-001_10_14_25.txt`** — Results examples
Examples of results from using these scripts for fine-tuning and generation experiments.

### **`/gpt2-files/`** — GPT-2 experiments
- **`generation/`**: Scripts for generating poetry in various forms (iambic pentameter, haiku) and interactive chat interfaces with model comparison
- **`tuning/`**: Fine-tuning scripts with partial layer unfreezing and line break preservation

### **`/llama-files/`** — Llama 3 experiments  
- **`generation/`**: Interactive poetry chat terminal for Llama 3 with comparison functionality
- **`tuning/`**: Fine-tuning scripts optimized for Llama 3 with 4-bit quantization

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

## 🛠️ **Tooltips / Code Glossary**

**🔧 LoRA** <sub>Low-Rank Adaptation: freezes base model weights and learns small rank-decomposition matrices for efficient fine-tuning.</sub>

**🎚️ Top-k / Top-p Sampling** <sub>Decoding strategies that control randomness by limiting token selection to the highest-probability candidates.</sub>

**🧱 Unfreezing Layers** <sub>Gradually enabling gradient updates in deeper transformer blocks to increase control without overfitting.</sub>

**🧮 Iambic Pentameter Metric** <sub>A custom evaluator that counts unstressed/stressed syllable alternation — an experiment in stylistic constraint scoring.</sub>

---

## 🗺️ **Project Roadmap**

- Add automated metric visualizations
- Add a full reproducible fine-tuning pipeline
- Add more evaluation scripts (e.g., perplexity tracking)
- Add a prompt-stability benchmarking suite
- Add LoRA-based adapters for Llama 3.1 8B

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
Technical Program Manager — AI Governance & Applied AI Operations  
📧 [mfbottari@gmail.com](mailto:mfbottari@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/michael-bottari)  
💻 [GitHub](https://github.com/bottari)

---

## 📝 **License**

MIT License — feel free to fork, study, and experiment.