import gradio as gr
import os

# Set this environment variable to disable the API endpoint discovery
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

def greet(name):
    return f"Hello {name}!"

# We define the demo as a variable
with gr.Blocks(analytics_enabled=False) as demo:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output")
    greet_btn = gr.Button("Submit")
    greet_btn.click(fn=greet, inputs=name, outputs=output)

if __name__ == "__main__":
    # We pass show_api=False here
    # We also use share=False to ensure no external tunnels attempt to 
    # validate the API schema via external network requests
    demo.launch(
        show_api=False, 
        server_name="0.0.0.0", 
        server_port=7860
    )
