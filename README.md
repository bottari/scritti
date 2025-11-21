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
- Custom evaluation metrics (e.g., early work on iambic pentameter scoring)  
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

## 📁 **Repository Structure**

```text
📁 scritti/
├── 📄 README.md
├── 📄 First_Edition_GenPs-001_10_14_25.txt
├── 📁 gpt2-files/
│   ├── 📁 generation/
│   │   ├── 📄 gpt2-generate-iambic-pentameter-003.py
│   │   ├── 📄 gpt2-generation-haiku_form-002.py
│   │   ├── 📄 interactive-poetry-chat-in-terminal-002--gpt2-with-comparison.py
│   │   └── 📄 the-gpt2-fine-tuning-script-thats-the-best-tweaked-002-unfreeze-top-layers--chatbot-compare.py
│   └── 📁 tuning/
│       └── 📄 the-gpt2-fine-tuning-script-thats-the-best-tweaked-003-unfreeze-top-layers---keep-source-line-breaks.py
└── 📁 llama-files/
    ├── 📁 generation/
    │   └── 📄 interactive-poetry-chat-in-terminal-for-llama3-002-with-comparison.py
    └── 📁 tuning/
        └── 📄 fine-tuning-script-for-llama-3-q4-001.py
```

### **`First_Edition_GenPs-001_10_14_25.txt`** — Results examples
Examples of results from using these scripts for fine-tuning experiments.

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

## 🔥 **Featured Work: GPT-2 Fine-Tuning (Before vs After)**

Below is a real output comparison between **GPT-2** and my **fine-tuned model**, showing my results. Please note the generated output is not optimized for narrative coherence, but instead it is optimized for stylistic control. (I prefer surreal and boderline nonsensical poetry.)

<details>
<summary><strong>🧠 BASE (GPT2) MODEL OUTPUT 🧠</strong> (click to expand)</summary>
<br/>
"bringing some frankenscence to the altar's work. (Perfect Right Now, Courtesy To He Named Abraham Smith)
For those interested in Jewry generally speaking, Uzzler described working on these murals-often perceived as near heavenly appearance by both Jewish and non-Jewish audiences who might not be familiar with Zarahemla. Steinberg also refers certain talents bearing family pedigree out of New York for honor lacing into his area: "We call him Mrs Millman here," he reports; we must don jeans if you're playing a painter because it surely counts at Old Miss Sestopol." Contrary to feeling that no giving is permitted when talking about rock n' roll sounds like salvageaging through trenches less than everybody else grinding roast garlic mustard right down their throats, Miroslav Pralunde doubtless saw Frank Iverson once speak during life unconstrained PR images or interviewing photographer Gus Schulze before closing your eyes wide-eyedly onto this project nonetheless. When things were cool outside Eugene Jackson Auditorium Hammerbuster Gooligan drummer Omar Tuni announced Gene Clancy would face crime but couldn't see two women driving far beyond Paddock Meadow...

…"The only thing saving me from having beer ten bucks bought" Then JACK"
</details>

<details>
<summary><strong>🎭 FINETUNED (POESIA DI MICHELE BOTTARI) MODEL OUTPUT 🎭</strong> (click to expand)</summary>
</br>
"bringing some frankenscence to the altar. you look at him from step number two. the red headed man stands in front of them. “This man’s taken refuge under an archway, being sworn to his father”s cave. come this day. we'll tell you why he was here for us: all too well. -“*s like it’s fine."” mx does not pause for a long breath. the room is covered with bones, feelings and dressings. he walks down the tramp-like path into his tent. jade fires over the campfire. tents are piled high on top huts; heavily padded fireplaces surround the spaces. asleep, white sheets place unzipped drinks onto a pile of cool sleeping bags. what moved Jade?” pensively. “lose the fucking thing?” he nods. he slowly moves closer between the benches, looking more closely at you. he stays where he is supposed to, so that you don't even come close in the moment, becoming fascinated. he pauses until you take notice. your entire body looks numb and exhausted. your lips begin sobbing"

</details>

---

## 🧰 **Key Technologies Used**

- Python (tooling, parsing, orchestration)
- PyTorch
- Transformers (Hugging Face)
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