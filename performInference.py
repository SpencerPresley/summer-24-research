from bitnet.inference import BitNetInference
import torch
# print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
inference_model = BitNetInference()
model_path = "bitnet_300M_final_model.pth"
inference_model.load_model(model_path)

input_text = "The history of computer science began long before the modern discipline of computer science that emerged in the 20th century. The progression from mechanical inventions and mathematical theories towards modern computer concepts and machines traces a history that forms a major part of the history of technology."
generated_length = 500
generated_text = inference_model.generate(input_text, generated_length)
print(f"Generated text:\n{generated_text}")