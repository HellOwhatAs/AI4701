from main import plate_recognition
import gradio as gr, cv2

def func(inputs):
    inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
    return plate_recognition(inputs)

demo = gr.Interface(func, inputs='image', outputs='text')
demo.launch(share=True)
