import gradio as gr

def greet(name):
    return f"Hello {name}!"

with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output")
    btn = gr.Button("Submit")
    btn.click(fn=greet, inputs=name, outputs=output)

if __name__ == "__main__":
    # We remove ssr=False. 
    # show_api=False is the most important fix for your previous error.
    demo.launch(
        show_api=False, 
        server_name="0.0.0.0", 
        server_port=7860
    )
