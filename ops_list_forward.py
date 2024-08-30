import onnx
import torch
import os
import tempfile
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

def analyze_torch_ops(model):
    op_count = Counter()

    def count_ops(module):
        for name, child in module.named_children():
            if list(child.children()):
                count_ops(child)
            else:
                op_count[child.__class__.__name__] += 1

    count_ops(model)
    return op_count

def export_to_onnx(model, tokenizer, model_name):
    onnx_folder = "onnx_models"
    os.makedirs(onnx_folder, exist_ok=True)

    onnx_filename = f"{model_name.split('/')[-1]}.onnx"
    onnx_path = os.path.join(onnx_folder, onnx_filename)
    
    inp = tokenizer("My cat is", return_tensors="pt").input_ids
    
    torch.onnx.export(
        model,
        inp,
        onnx_path,
        opset_version=18,
        do_constant_folding=False,
        input_names=['input_ids'],
        output_names=['output'],
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 'output': {0: 'batch_size', 1: 'sequence'}},
        export_params=False,
        training=torch.onnx.TrainingMode.EVAL,
        keep_initializers_as_inputs=False,
    )
    
    print(f"ONNX model saved to: {onnx_path}")
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    return onnx_path

def analyze_llama_model(model_name, auth_token):
    with tempfile.TemporaryDirectory() as tmpdir:
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=auth_token, cache_dir=tmpdir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=auth_token, cache_dir=tmpdir)
        
        print(model_name)
        print('----------')
        print(model)

        torch_ops = analyze_torch_ops(model)
    
    with torch.no_grad():
        onnx_path = export_to_onnx(model, tokenizer, model_name)

    model_p = onnx.load(onnx_path)
    graph = model_p.graph
    onnx_ops = Counter(node.op_type for node in graph.node)
    
    return torch_ops, onnx_ops

def save_to_csv(data, filename):
    df = pd.DataFrame(data).fillna(0).astype(int)
    df.index.name = 'Operation'
    df.to_csv(filename)

auth_token = "-----"

models = {
    "Llama-3_1-8b": "meta-llama/Meta-Llama-3.1-8B",
    "Llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "Llama-2-7b": "meta-llama/Llama-2-7b-hf",
}

torch_data = {}
onnx_data = {}

for model_name, model_path in models.items():
    print(f"Analyzing {model_name}:")
    torch_ops, onnx_ops = analyze_llama_model(model_path, auth_token)
    torch_data[model_name] = torch_ops
    onnx_data[model_name] = onnx_ops
    print(f"PyTorch ops: {sum(torch_ops.values())}")
    print(f"ONNX ops: {sum(onnx_ops.values())}")
    print(f"ONNX graph saved as {model_name.split('/')[-1]}.onnx")
    print()

save_to_csv(torch_data, "torch_ops.csv")
save_to_csv(onnx_data, "onnx_ops.csv")
