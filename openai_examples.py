#!/usr/bin/env python3
"""
OpenAI API Examples

This module demonstrates various OpenAI API functionalities using the OpenAIClient utility.
Run this file to see examples of:
- Chat completions
- Text completions
- Image generation
- Text embeddings
- Interactive chat mode
"""

import sys
from openai_app import OpenAIClient


def main():
    """Main function demonstrating various OpenAI API calls"""
    
    try:
        # Initialize the OpenAI client
        ai_client = OpenAIClient()
        print("✅ OpenAI client initialized successfully!\n")
        
        # Example 1: Simple chat completion
        print("🤖 Example 1: Simple Chat Completion")
        print("-" * 50)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        response = ai_client.chat_completion(messages)
        print(f"Response: {response.choices[0].message.content}\n")
        
        # Example 2: Creative writing with higher temperature
        print("📝 Example 2: Creative Writing")
        print("-" * 50)
        
        creative_messages = [
            {"role": "system", "content": "You are a creative storyteller."},
            {"role": "user", "content": "Write a short story about a robot learning to paint."}
        ]
        
        creative_response = ai_client.chat_completion(
            messages=creative_messages,
            temperature=1.0,
            max_tokens=200
        )
        print(f"Story: {creative_response.choices[0].message.content}\n")
        
        # Example 3: Text completion (legacy endpoint)
        print("📄 Example 3: Text Completion")
        print("-" * 50)
        
        prompt = "The benefits of artificial intelligence include"
        completion_response = ai_client.text_completion(prompt, max_tokens=100)
        print(f"Completion: {completion_response.choices[0].text.strip()}\n")
        
        # Example 4: Generate embeddings
        print("🔢 Example 4: Text Embeddings")
        print("-" * 50)
        
        text_to_embed = "OpenAI provides powerful AI models for various applications."
        embeddings_response = ai_client.get_embeddings(text_to_embed)
        embedding_vector = embeddings_response.data[0].embedding
        print(f"Embedding dimension: {len(embedding_vector)}")
        print(f"First 5 values: {embedding_vector[:5]}\n")
        
        # Example 5: Image generation (commented out by default)
        # print("🎨 Example 5: Image Generation")
        # print("-" * 50)
        # 
        # image_prompt = "A serene mountain landscape with a crystal clear lake reflecting the snow-capped peaks"
        # image_response = ai_client.generate_image(image_prompt)
        # print(f"Generated image URL: {image_response.data[0].url}\n")
        
        # Example 6: Conversation with multiple exchanges
        print("💬 Example 6: Multi-turn Conversation")
        print("-" * 50)
        
        conversation = [
            {"role": "system", "content": "You are a knowledgeable tutor."},
            {"role": "user", "content": "Can you explain what machine learning is?"},
        ]
        
        # First response
        response1 = ai_client.chat_completion(conversation)
        assistant_reply = response1.choices[0].message.content
        print(f"Assistant: {assistant_reply}")
        
        # Add to conversation and continue
        conversation.append({"role": "assistant", "content": assistant_reply})
        conversation.append({"role": "user", "content": "Can you give me a simple example?"})
        
        response2 = ai_client.chat_completion(conversation)
        print(f"Assistant: {response2.choices[0].message.content}\n")
        
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Installed required packages: pip install openai python-dotenv")
        print("2. Set your OpenAI API key in a .env file or OPENAI_API_KEY environment variable")


def interactive_chat():
    """Interactive chat function for real-time conversation"""
    try:
        ai_client = OpenAIClient()
        print("🤖 Interactive Chat with OpenAI")
        print("Type 'quit' to exit, 'clear' to clear conversation history\n")
        
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            elif user_input.lower() == 'clear':
                conversation = [{"role": "system", "content": "You are a helpful assistant."}]
                print("Conversation cleared! 🧹\n")
                continue
            elif not user_input:
                continue
            
            conversation.append({"role": "user", "content": user_input})
            
            try:
                response = ai_client.chat_completion(conversation)
                assistant_reply = response.choices[0].message.content
                print(f"Assistant: {assistant_reply}\n")
                conversation.append({"role": "assistant", "content": assistant_reply})
            except Exception as e:
                print(f"Error: {e}\n")
    
    except Exception as e:
        print(f"Failed to initialize chat: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--chat":
        interactive_chat()
    else:
        main()
