import gradio as gr

def test(text):
    return f"Success: {text}"

# We explicitly turn off every feature that triggers API schema generation
demo = gr.Interface(
    fn=test,
    inputs="text",
    outputs="text",
    analytics_enabled=False
)

if __name__ == "__main__":
    demo.launch(
        show_api=False, 
        ssr=False, 
        enable_monitoring=False,
        server_name="0.0.0.0", 
        server_port=7860
    )
