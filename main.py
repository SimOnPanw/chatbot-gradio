import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "cognitivecomputations/dolphin-2.8-mistral-7b-v02"

try:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer: {e}")

# Initialize conversation history
history = []

def generate_response(user_input, history):
    try:
        # Append the user input to the history
        history.append({"role": "user", "content": user_input})
        
        # Prepare the context by concatenating the conversation history
        context = " ".join([f"{item['role']}: {item['content']}" for item in history])
        
        # Tokenize the context
        inputs = tokenizer(context, return_tensors="pt")
        
        # Ensure inputs are on the same device as the model
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        
        # Generate the response
        outputs = model.generate(**inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Append the model's response to the history
        history.append({"role": "bot", "content": response})
        
        # Return the response and the updated history
        return response, history
    except torch.cuda.OutOfMemoryError:
        return "Model ran out of memory. Please try a shorter input.", history
    except Exception as e:
        return f"An error occurred: {e}", history

# Set up the Gradio interface
interface = gr.Interface(
    fn=generate_response,
    inputs=[gr.Textbox(lines=2, placeholder="Enter your message here...")],
    outputs=[gr.Textbox(lines=10)],
    live=True,
    title="Dolphin Mistral Chat Bot",
    description="A chatbot powered by the Dolphin Mistral model.",
    theme="default"
)

if __name__ == "__main__":
    try:
        interface.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        print(f"Failed to launch the Gradio interface: {e}")
