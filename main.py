import torch

from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

import gradio as gr
import argparse
import os

# TODO(mrsteyk): better fucking docstrings and structure lmfao

def generate_prelude(system: str, sep: str, system1="Assistant is a distilled language model trained by the community.", system2="", prelude="quality: high"):
    """
        Generates prelude. Defaults do not mimick OpenAI completely. `system1` is official prefix, `system2` is additional prefix.
    """
    return f"{prelude}\n\n[{system}]\n{system1}{sep}\n\n[{system}]\n{system2}{sep}\n\n"

# RN idfk how one would integrate System entity into all of this bs
# TODO(mrsteyk): add more flexibility on who's the starting entity?
def generate_history_raw(user: str, ai: str, sep: str, history: List[Tuple[str, str]]):
    """
        Low Level history generation. No pruning is done here.
    """
    return '\n\n'.join([f"[{user}]\n{u}{sep}\n\n[{ai}]\n{a}{sep}" for (u, a) in history]) + '\n\n' if len(history) > 0 else ''

def prune_history(history: List[Tuple[str, str]], max_exchanges=5):
    """
        Truncates the history down to first + `max_echanges`. `max_exchanges=5` mimicks OpenAI.
    """
    if len(history) <= (max_exchanges + 1):
        return history
    else:
        return [history[0]] + history[-max_exchanges:]

def generate_history(user: str, ai: str, sep: str, history: List[Tuple[str, str]], max_echanges=5):
    """
        High level history generation. Defaults mimick OpenAI.
    """
    return generate_history_raw(user, ai, sep, prune_history(history, max_exchanges=max_echanges))

def generate_query(inpt: str, user: str, ai: str, system: str, sep: str, history: List[Tuple[str, str]], max_echanges=5):
    # TODO(mrsteyk): more flexibility
    prelude = generate_prelude(system, sep)
    hist = generate_history(user, ai, sep, history, max_echanges=max_echanges)
    # TODO(mrsteyk): prune user message
    return f"{prelude}{hist}[{user}]\n{inpt}{sep}\n\n[{ai}]\n"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="A model to load")
    parser.add_argument("--debug", action='store_true', help="Debug shit")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model)
    if torch.cuda.is_available():
        model.half().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hist = []

    def prune_history_gradio():
        global hist
        hist = prune_history(hist)
        return hist.copy()
    
    def remove_last_exchange_gradio():
        global hist
        if len(hist) > 0:
            hist.pop()
        return hist.copy()
    
    def clear_history():
        global hist
        hist = []
        return []

    def generate(inpt, system, user, ai, max_new_tokens):
        global model, tokenizer, hist
        query = generate_query(inpt, user, ai, system, tokenizer.sep_token, hist)
        print(query)
        tokens = tokenizer(query, return_tensors="pt").input_ids
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        # TODO(mrsteyk): parameters
        res = model.generate(tokens,
            max_new_tokens=max_new_tokens, do_sample=True, top_k=1, top_p=0.9, eos_token_id=tokenizer.sep_token_id, temperature=1.0)
        res = tokenizer.batch_decode([i[len(tokens[0]):] for i in res], skip_special_tokens=True)[0]
        hist.append((inpt, res))
        return hist.copy()
    
    css=''
    if os.path.exists("user.css"):
        with open("user.css", "r") as f:
            css = f.read()
    with gr.Blocks(analytics_enabled=False, css=css) as interface:
        gr.Markdown("# OpenChatGPT\n\nA simple demo of **ChatGPT**-like backend...")
        with gr.Column(variant="compact"):
            display = gr.Chatbot()
            with gr.Row():
                with gr.Column(scale=7):
                    inpt = gr.Textbox(lines=2, placeholder="Your message", show_label=False)
                submit = gr.Button("Send", variant="primary")
            history_prune = gr.Button("Prune history")
            remove_last_exchange = gr.Button("Remove last exchange")
            clear = gr.Button("Clear history")
        
        system = gr.Textbox(value="System", lines=1, label="System's name")
        user = gr.Textbox(value="User", lines=1, label="Your name")
        ai = gr.Textbox(value="Assistant", lines=1, label="AI's name")

        max_new_tokens = gr.Slider(767, 2048, step=1, label="Max new tokens to generate")

        submit.click(fn=generate,
            inputs=[inpt, system, user, ai, max_new_tokens],
            outputs=display)
        history_prune.click(fn=prune_history_gradio, outputs=display)
        remove_last_exchange.click(fn=remove_last_exchange_gradio, outputs=display)
        clear.click(fn=clear_history, outputs=display)
    
    interface.queue()
    interface.launch()