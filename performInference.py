from bitnet.inference import BitNetInference
import torch
# print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
inference_model = BitNetInference()
model_path = "bitnet_300M_final_model.pth"
inference_model.load_model(model_path)

input_text = "In physics, the conservation laws state that certain physical properties (such as energy, mass, momentum, and angular momentum) of an isolated system remain constant over time. These laws are fundamental to the understanding of many physical processes, and they are often derived from the symmetries of the physical system."
generated_length = 1024
generated_text = inference_model.generate(input_text, generated_length)
print(f"\nInput text:\n{input_text}")
print(f"\nGenerated text:\n{generated_text}")