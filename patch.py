with open("backend/model_manager.py", "r") as f:
    content = f.read()

# I will replace the previously patched Qwen 2.5/3.5 entries with the real Qwen3.5-35B-A3B from unsloth.
content = content.replace("qwen2.5-32b-instruct.Q4_K_M", "Qwen3.5-35B-A3B-UD-Q4_K_M")
content = content.replace("qwen3.5-32b-instruct.Q4_K_M", "Qwen3.5-35B-A3B-UD-Q4_K_M")
content = content.replace("Qwen 2.5 32B Instruct", "Qwen 3.5 35B A3B")
content = content.replace("Qwen 3.5 32B Instruct", "Qwen 3.5 35B A3B")
content = content.replace("Alibaba's advanced 32B model. Highly capable in reasoning, coding, and multilingual tasks.", "Alibaba's latest Qwen3.5 35B model. Highly capable and efficient.")
content = content.replace("Alibaba advanced 32B model. Highly capable in reasoning, coding, and multilingual tasks.", "Alibaba's latest Qwen3.5 35B model. Highly capable and efficient.")
content = content.replace("https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf", "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-Q4_K_M.gguf")

with open("backend/model_manager.py", "w") as f:
    f.write(content)
