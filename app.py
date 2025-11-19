# src/app.py
import gradio as gr
from main import run_pipeline

def ask(query):
    res = run_pipeline(query)
    if res is None:
        return "No results", None
    return res["answer"], res["table"].to_dict(orient="records")

with gr.Blocks() as demo:
    gr.Markdown("# AI Web Search + RAG")
    inp = gr.Textbox(label="Query")
    btn = gr.Button("Search")
    out_txt = gr.Textbox(label="Answer")
    out_table = gr.Dataframe(headers=["title","url","snippet","query"])

    btn.click(fn=ask, inputs=inp, outputs=[out_txt, out_table])

if __name__ == "__main__":
    demo.launch()
