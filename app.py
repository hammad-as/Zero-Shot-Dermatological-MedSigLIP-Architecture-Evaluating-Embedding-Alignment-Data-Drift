import gradio as gr

def process(input_data):
    # This is your placeholder for your actual model logic
    return f"Processed: {input_data}"

# Use gr.Interface: it handles the API and UI generation 
# without needing the complex, error-prone 'Blocks' configuration.
demo = gr.Interface(
    fn=process,
    inputs=gr.Textbox(label="Input"),
    outputs=gr.Textbox(label="Output"),
    title="MedSigLIP Dashboard"
)

# Do NOT pass complex arguments to launch()
# Keep it as minimal as possible to avoid ASGI/Middleware conflicts.
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
