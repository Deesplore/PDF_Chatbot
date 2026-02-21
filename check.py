from google import genai

client = genai.Client(
    api_key=google_api_key
)

models = client.models.list()

for m in models:
    print(m.name)

