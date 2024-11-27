import openai

openai.api_key=""
#""



if __name__ == '__main__':
    print("start...")
    message = input("")
    messages = []
    while message != "quit":
        print("type your messages:")
        message = input("")
        if message != "quit":
            messages.append({"role": "user", "content": message})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            reply = response["choices"][0]["message"]["content"]
        print(reply)