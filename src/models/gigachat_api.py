import json
import time

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

with open("./models/gigachat_secrets_2.json", "r", encoding="utf-8") as f:
    gigachat_secrets = json.load(f)


# pip install gigachat
class GIGACHAT_cstm:

    def __init__(self):
        self.giga = GigaChat(
            credentials=gigachat_secrets["auth_key"],
            verify_ssl_certs=False
        )
        response_acces_token = self.giga.get_token()
        self.access_token = response_acces_token.access_token
        self.models_list = [x.id_ for x in self.giga.get_models().data]
        print("Available models")
        [print(x) for x in self.models_list]
        self.chosen_model = "GigaChat-2-Max"

    def process(self, messages):
        gigachat_messages_list = [
            Messages(role=MessagesRole(x["role"]), content=x["content"]) for x in messages
        ]
        response_giga_model = self.giga.chat(
            Chat(
                # model="GigaChat-2-Max",
                model=self.chosen_model,
                messages=gigachat_messages_list
            )
        )
        return response_giga_model.choices[0].message.content


if __name__ == "__main__":
    giga_cstm_instance = GIGACHAT_cstm()
    start_time = time.time()
    result_text = giga_cstm_instance.process([{"role": "user", "content": "Посчитай от одного до десяти"}])
    print(f"GPT TIME TAKE: {time.time() - start_time} s.")
    print(f"GIGACHAT> {result_text}")
