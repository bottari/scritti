import torch
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = r"D:\models\gemma4-poetry-finetune-whitman\merged_fp16",
    max_seq_length = 2048,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)

prompt_messages = [{"role": "user", "content": [{"type": "text", "text": "what is the meaning of this?"}]}]
input_ids = tokenizer.apply_chat_template(
    prompt_messages,
    tokenize              = True,
    add_generation_prompt = True,
    return_tensors        = "pt",
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens = 256,
        temperature    = 0.5,
        top_p          = 0.95,
        do_sample      = True,
    )

generated = output_ids[0][input_ids.shape[-1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))