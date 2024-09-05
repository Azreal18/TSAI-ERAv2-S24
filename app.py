import gradio as gr

from Stable_difusion import (
    stl_list,
    img_size_opt_dict,
    loss_fn_dict,
    generate_images,
)

with gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Textbox(label="Enter prompt text here"),
        gr.Slider(
            minimum=1,
            maximum=50,
            step=1,
            default=30,
            label="Number of inference steps",
            description="Choose between 1 and 50",
        ),
        gr.MultiDropdown(
            options=stl_list,
            label="Style",
            description="Styles to be applied on images",
        ),
        gr.Dropdown(
            options=img_size_opt_dict,
            label="Image size",
            description="Target size for generated images",
        ),
        gr.Radio(
            options=loss_fn_dict,
            label="Additional guidance loss",
            description="The loss to be applied",
        ),
        gr.Textbox(
            label="Enter additional guidance text here if text-image similarity loss is selected",
        ),
    ],
    outputs=[
        gr.Image(label="Without guidance", shape=(300, 300)),
        gr.Image(label="With guidance", shape=(300, 300)),
    ],
    title="Stable Diffusion - Textual Inversion and additional guidance",
    description="Generates images based on the prompt and 5 different styles and then with additional guidance",
    article="The image generation may take 5 to 10 minutes on CPU per image",
    examples=[
        ["A cat"],
        ["A puppy"],
    ],
) as app:
    app.launch()
