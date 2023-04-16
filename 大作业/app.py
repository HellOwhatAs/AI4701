from main import plate_recognition
import gradio as gr, cv2

def func(inputs):
    if inputs is None: return [], ''
    inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
    ret = []
    try:
        for tmp in plate_recognition(inputs, show_intermediate=True):
            ret.append(tmp)
        return ret[:-1], ret[-1]
    except:
        return ret, '识别失败了！'
    

demo = gr.Interface(
    func, 
    inputs = gr.Image(label='输入车牌号图像'),
    outputs = [
        gr.Gallery(label='中间值').style(columns=[3], object_fit="contain", height="auto"), 
        gr.Textbox(label='识别结果')
    ],
    allow_flagging = 'never',
    examples = [ 
        './resources/images/easy/1-1.jpg',
        './resources/images/easy/1-2.jpg',
        './resources/images/easy/1-3.jpg',
        './resources/images/medium/2-1.jpg',
        './resources/images/medium/2-2.jpg',
        './resources/images/medium/2-3.jpg',
        './resources/images/difficult/3-1.jpg',
        './resources/images/difficult/3-2.jpg',
        './resources/images/difficult/3-3.jpg'
    ],
    title = '计算机视觉期末课程大作业'
)
demo.launch(inbrowser=True)