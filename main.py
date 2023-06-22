import openai
import os

skey = os.getenv("oai_skey")
okey = os.getenv("oai_okey")

openai.api_key = skey
openai.organization = okey
resp = openai.Model.list()

model = "gpt-3.5-turbo"

messages = [{"role": "system", "content": "Do you know how I am sending this request?"}]

response = openai.ChatCompletion.create(model = model,messages = messages)

print(response)

text_file = open("response.txt","w")
text_file.write(str(response))
text_file.close()
