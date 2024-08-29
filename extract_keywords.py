import os

from openai import OpenAI as ClientOpenAI
from openai_integration import OpenAI, KeyLLM

ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")
PROJECT = os.getenv("OPENAI_PROJECT")
API_KEY = os.getenv("OPENAI_API_KEY")

client = ClientOpenAI(
    organization=ORGANIZATION,
    project=PROJECT,
    api_key=API_KEY,
)

prompt_chat = """
У мене є такий документ:
[DOCUMENT]

На основі наведеної вище інформації виділіть ключові слова, які найкраще описують тему тексту.
Ключові слова повинні бути не більше двох слів.

Використовуйте наступний формат, розділений комами:
<keywords>
""".strip()

llm = OpenAI(
    client,
    system_prompt="Ви корисний помічник.",
    prompt=prompt_chat,
    exponential_backoff=True,
    # model="gpt-4o-mini",
    model="gpt-4o-2024-08-06",
    generator_kwargs={"max_tokens": 128},
)

kw_model = KeyLLM(llm)

document = """
Суміжні країни повернулися до правила, що для перетину кордону українці повинні мати закордонний паспорт.

Як пояснив голова Держприкордонслужби, якщо людина перетинає кордон як біженець – прямуючи перш за все з районів бойових дій – і має об'єктивні підстави, що не може поновити швидко або зробити закордонний паспорт, "тоді у виключних випадках суміжна сторона може пропустити громадян України за внутрішніми паспортами".

Українські прикордонники при цьому вивчають обставини виїзду і можуть контактувати з колегами суміжної країни, щоб зрозуміти, чи вони пропустять людину за таких обставин за внутрішнім паспортом.
"""

keywords = kw_model.extract_keywords(document)

cleaned_keywords = [
    keyword.replace("<keywords>", "").replace("</keywords>", "") for keyword in keywords
]
lowercased_keywords = [keyword.lower() for keyword in cleaned_keywords]

print(lowercased_keywords)
