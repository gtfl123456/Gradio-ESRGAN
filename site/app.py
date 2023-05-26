import os
import numpy as np
import gradio as gr
from PIL import Image
from inference_realesrgan import main

Rpath = os.path.dirname(os.path.realpath(__file__))#获取文件的绝对路径

def save_pic(img,path):#服务与produce
    img = Image.fromarray(img)
    img.save(path)

def produce(x):
    pre_path = "./inputs"
    out_path = "./outputs"
    i = len(os.listdir(pre_path))
    path = pre_path+"/"+f"{i}.png"#根据图片的路径进行保存
    save_pic(x, path)  # 保存上传的图片
    main(path)
    out_img = out_path+"/"+f"{i}_out.png"
    out_img = np.array(Image.open(out_img))
    return out_img#返回一个numpy格式的图片

with gr.Blocks() as demo:
    gr.Markdown("基于Gradio的Real-ESRGAN超分项目演示.")
    with gr.Tab("Flip Image"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("Submit")

    with gr.Accordion("Open for More!"):
        gr.Markdown("gtteam")

    image_button.click(fn=produce, inputs=image_input, outputs=image_output)

demo.launch()