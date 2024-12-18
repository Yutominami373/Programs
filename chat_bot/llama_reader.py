import ollama

class AI_Assistant:
    def __init__(self):
        self.full_transcript = [
            {"role":"system", "content":"あなたは間違いを日本語で教えるAI講師です"}
        ]

    def receive_user_input(self, user_input):
        #print(f"User: {user_input}")
        self.full_transcript.append({"role": "user", "content": user_input})
        return self.generate_ai_response()

    def generate_ai_response(self):
        response_stream = ollama.chat(
            model="notpotato",
            messages=self.full_transcript,
            stream=True
        )
        
        text_buffer = ""
        full_text = ""
        for chunk in response_stream:
            text_buffer += chunk['message']['content']
            if  text_buffer.endswith('。'):
                #print(text_buffer, end="\n", flush=True)
                full_text += text_buffer
                text_buffer = ""
        if text_buffer:
            #print(text_buffer, end="\n", flush=True)
            full_text += text_buffer
        self.full_transcript.append({"role":"assistant", "content":full_text})
        
        #print("fulltext: ",full_text)
        return full_text
    



