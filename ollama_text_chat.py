#!/usr/bin/env python3
"""Simple text-based chat with Ollama."""

import ollama

def chat():
    # User requested qwen3:4b-instruct specifically
    model = "qwen3:4b-instruct"
    messages = []
    
    print(f"Starting chat with {model}...")
    print("Type 'quit', 'exit', or Ctrl+C to end.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            # Add user message to history
            messages.append({'role': 'user', 'content': user_input})
            
            print("Assistant: ", end="", flush=True)
            
            try:
                # Stream the response
                stream = ollama.chat(
                    model=model,
                    messages=messages,
                    stream=True
                )
                
                full_response = ""
                for chunk in stream:
                    content = chunk['message']['content']
                    print(content, end="", flush=True)
                    full_response += content
                
                print() # Newline after response
                
                # Add assistant response to history
                messages.append({'role': 'assistant', 'content': full_response})
                
            except ollama.ResponseError as e:
                print(f"\nOllama Error: {e}")
                if e.status_code == 404:
                    print(f"Model '{model}' not found. Try running: ollama pull {model}")
            except Exception as e:
                print(f"\nError communicating with Ollama: {e}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    chat()
