"""Run this model in Python

> pip install azure-ai-inference
"""
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage, ToolMessage
from azure.ai.inference.models import ImageContentItem, ImageUrl, TextContentItem
from azure.core.credentials import AzureKeyCredential

from dotenv import load_dotenv
load_dotenv(override=True)

# To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings.
# Create your PAT token by following instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
client = ChatCompletionsClient(
    endpoint = "https://models.github.ai/inference",
    credential = AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
    api_version = "2024-12-01-preview",
)

messages = [
    SystemMessage(content = "Sei un assistente che crea contenuti per i social media nei seguenti formati: blog, script per video e didascalie per i post. \n# Formati di Contenuto: \n- **Blog**: Articoli informativi o narrativi, strutturati con introduzione, corpo e conclusione. \n- **Script per Video**: Testo per video, con indicazioni per le scene, i dialoghi e le azioni.\n- **Didascalie per Post**: Testi brevi e accattivanti per accompagnare immagini o video sui social media.\n# Stile\nUsa uno stile informale che sia adatto ad un pubblico di sviluppatori. Includi emoticons e hashtags. \n"),
    UserMessage(content = [
        TextContentItem(text = "Crea un post per social media che promuova un nuovo strumento di intelligenza artificiale per sviluppatori."),
    ]),
]

while True:
    response = client.complete(
        messages = messages,
        model = "mistral-ai/mistral-medium-2505",
        temperature = 0.8,
        top_p = 0.1,
    )

    if response.choices[0].message.tool_calls:
        print(response.choices[0].message.tool_calls)
        messages.append(response.choices[0].message)
        for tool_call in response.choices[0].message.tool_calls:
            messages.append(ToolMessage(
                content=locals()[tool_call.function.name](),
                tool_call_id=tool_call.id,
            ))
    else:
        print(f"[Model Response] {response.choices[0].message.content}")
        break
