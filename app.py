import gradio as gr
from utils import predict_animal
from PIL import Image

def classify_interface(image):
    return predict_animal(image)

# Custom CSS for centering and polishing
custom_css = """
.container { max-width: 900px; margin: auto; }
.title { text-align: center; margin-bottom: 20px; }
.footer { text-align: center; margin-top: 20px; font-size: 0.8em; color: gray; }
"""

# Implement a professional theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)

# Define the Gradio interface
with gr.Blocks(theme=theme, css=custom_css, title="Animal Image Classifier") as demo:
    
    with gr.Column(elem_classes="container"):
        # Header
        gr.Markdown(
            """
            # 🦁 Animal Image Classifier
            ### Identify animal species with the power of EfficientNetB0
            """,
            elem_classes="title"
        )
        
        with gr.Row():
            # Left Column: Input and Examples
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil", 
                    label="Upload Image", 
                    sources=["upload", "clipboard"], 
                    height=300
                )
                
                with gr.Row():
                    clear_btn = gr.ClearButton(components=[input_image], variant="secondary")
                    submit_btn = gr.Button("Classify Animal", variant="primary")
                
                # Examples under the input for easy testing
                gr.Examples(
                    examples=["examples/bird.png", "examples/lion.jpg", "examples/sheep.jpg", "examples/dog (2).jpeg","examples/spider.jpg"], 
                    inputs=input_image, 
                    label="Try these examples"
                )

            # Right Column: Output
            with gr.Column(scale=1):
                output_label = gr.Label(num_top_classes=3, label="Predictions")
                
                # Instructions in an Accordion to keep UI clean
                with gr.Accordion("ℹ️ How it works", open=False):
                    gr.Markdown(
                        """
                        1. **Upload** an image of an animal (or select an example).
                        2. **Click 'Classify Animal'**.
                        3. The model (**EfficientNetB0**) will analyze the image.
                        4. Results will show the top **animal categories** and their confidence scores.
                        """
                    )

        # Event listeners
        submit_btn.click(fn=classify_interface, inputs=input_image, outputs=output_label)
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("Created using [Gradio](https://gradio.app) and PyTorch", elem_classes="footer")

if __name__ == "__main__":
    demo.launch()
