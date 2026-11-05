# Simplest possible test: send one question to the local Ollama model
# and print its reply. If this works, Ollama and the model are set up correctly.
import ollama

# Ask the model a question. `messages` is the conversation: here just one user message.
answer = ollama.chat(
    model = "qwen3.5:9b",
    messages=[{"role": "user", "content": "Tell me what is agent in one sentence"}],
)

# The reply text lives at answer["message"]["content"].
print(answer["message"]["content"])
