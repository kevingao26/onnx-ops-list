import onnx
import onnx.helper

onnx_export_path = "onnx_models/Meta-Llama-3-8B.onnx"
model = onnx.load(onnx_export_path)
print(onnx.helper.printable_graph(model.graph))
