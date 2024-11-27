import openai
openai.api_key=""

response = openai.Image.create(
  prompt="快乐大本营",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']
print(image_url)