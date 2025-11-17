import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel

# Configuration
MODEL_NAME = "gpt2"
ADAPTER_PATH = "gpt2-finetuned-poetry-mercury-04/final_model"

def initialize_model():
    """Initialize and load both the base model and the PEFT finetuned model."""
    print("🔄 Loading model and tokenizer...")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load tokenizer (shared by both models)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 2. Load the BASE model (standard GPT-2)
    print("   -> Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base_model.to(device)
    base_model.eval()
    
    # 3. Load the FINETUNED model
    print("   -> Loading finetuned model (PEFT adapter)...")
    # Load a fresh instance of the base model to apply the adapter to
    finetuned_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Load the PEFT adapter and merge it with the base model
    finetuned_model = PeftModel.from_pretrained(finetuned_base, ADAPTER_PATH)
    finetuned_model = finetuned_model.merge_and_unload() # Merge weights for better performance
    finetuned_model.to(device)
    finetuned_model.eval()
    
    print(f"✅ Both models loaded successfully on {device}")
    # Return both models, the tokenizer, and the device
    return base_model, finetuned_model, tokenizer, device

def stream_generate(model, tokenizer, device, prompt, max_length, streamer):
    """
    Generates text from a given prompt and streams the output to the console.
    This function uses the TextStreamer for token-by-token output.
    
    FIX: Now uses tokenizer() to generate and pass the attention_mask 
    to prevent the related warning message.
    """
    # Use tokenizer() which returns both input_ids and attention_mask
    encoded_input = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text. The streamer handles the output.
    with torch.no_grad():
        model.generate(
            encoded_input['input_ids'],
            attention_mask=encoded_input['attention_mask'], # Pass the attention mask
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=0,
            top_p=0.95,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer # Key parameter for streaming
        )

def print_banner():
    """Print a welcome banner"""
    print("\n" + "="*80)
    print("🎭 POESIA DI MICHELE BOTTARI GENERATOR COMPARATOR 🎭")
    print("Comparing PEFT-Finetuned Model vs. Base Model (GPT2)")
    print("="*80)
    print("Welcome! Type your poetry prompts and I'll continue them, showing both results.")
    print("Commands:")
    print("  • Type any text to generate poetry")
    print("  • '/length <number>' to set max output length (e.g., '/length 150')")
    print("  • 'prompt /length <number>' to set length and generate (e.g., 'death /length 50')")
    print("  • 'quit' or 'exit' to stop")
    print("  • 'clear' to clear the screen")
    print("  • 'help' to show this message again")
    print("="*80 + "\n")

def clear_screen():
    """Clear the terminal screen"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """Main interactive chat loop"""
    try:
        # Initialize models (returns base_model, finetuned_model, tokenizer, device)
        base_model, finetuned_model, tokenizer, device = initialize_model()
        
        # Create the streamer: skip_prompt=True ensures only the generated text is streamed
        streamer = TextStreamer(
            tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True, 
            file=sys.stdout
        )
        
        # Default settings
        current_max_length = 250
        
        # Show banner
        print_banner()
        print(f"Current max length: {current_max_length} tokens\n")
        
        while True:
            try:
                # Get user input
                prompt = input("🖋️  Enter your poetry prompt: ").strip()
                
                actual_prompt = prompt
                
                # Command parsing logic (similar to your original script)
                if ' /length ' in prompt.lower():
                    parts = prompt.split(' /length ')
                    if len(parts) == 2:
                        actual_prompt = parts[0].strip()
                        try:
                            new_length = int(parts[1].strip())
                            if 10 <= new_length <= 1000:
                                current_max_length = new_length
                                print(f"✅ Max length set to {current_max_length} tokens")
                            else:
                                print("❌ Length must be between 10 and 1000 tokens\n")
                                continue
                        except ValueError:
                            print("❌ Invalid length number\n")
                            continue
                
                # Handle standalone commands
                if prompt.lower() in ['quit', 'exit', 'q', '/quit', '/exit', '/q']:
                    print("\n👋 Goodbye! Thanks for using the poetry generator!")
                    break
                elif prompt.lower() in ['clear', '/clear']:
                    clear_screen()
                    print_banner()
                    print(f"Current max length: {current_max_length} tokens\n")
                    continue
                elif prompt.lower() in ['help', '/help']:
                    print_banner()
                    print(f"Current max length: {current_max_length} tokens\n")
                    continue
                elif prompt.lower().startswith('/length ') or prompt.lower().startswith('length '):
                    try:
                        length_part = prompt.lower().replace('/length ', '').replace('length ', '')
                        new_length = int(length_part)
                        if 10 <= new_length <= 1000:
                            current_max_length = new_length
                            print(f"✅ Max length set to {current_max_length} tokens\n")
                        else:
                            print("❌ Length must be between 10 and 1000 tokens\n")
                        continue
                    except (IndexError, ValueError):
                        print("❌ Usage: /length <number> (e.g., '/length 150')\n")
                        continue
                elif not actual_prompt:
                    print("❌ Please enter a prompt or command.")
                    continue
                
                print(f"\n📝 Generating results for: '{actual_prompt}' (max {current_max_length} tokens)...")
                
                # -------------------------------------------------------------------
                # 1. FINETUNED MODEL GENERATION (Poetry)
                # -------------------------------------------------------------------
                print("\n" + "="*20 + " 🎭 FINETUNED (POESIA DI MICHELE BOTTARI) MODEL RESPONSE 🎭 " + "="*20)
                # Manually print the prompt before streaming starts
                sys.stdout.write(actual_prompt)
                sys.stdout.flush()
                stream_generate(finetuned_model, tokenizer, device, actual_prompt, current_max_length, streamer)
                
                # -------------------------------------------------------------------
                # 2. BASE MODEL GENERATION (General GPT2)
                # -------------------------------------------------------------------
                print("\n\n" + "="*20 + " 🧠 BASE (GPT2) MODEL RESPONSE 🧠 " + "="*20)
                # Manually print the prompt before streaming starts
                sys.stdout.write(actual_prompt)
                sys.stdout.flush()
                stream_generate(base_model, tokenizer, device, actual_prompt, current_max_length, streamer)
                
                print("\n" + "="*80 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during generation: {e}")
                print("Please try again with a different prompt.\n")
                
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        print("Please check your model path and dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()