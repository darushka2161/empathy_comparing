# architectures/empathy_zero_shot_short.py
# Архитектура: ZeroShot с ограничением длины (≤15 слов)
# Соответствует среднему стилю ответов в датасете EmpatheticDialogues
# LLM-вызовов: 1

SYSTEM_PROMPT = (
    "This is an empathetic dialogue task: The first worker (Speaker) is given an "
    "emotion label and writes his own description of a situation when he has felt "
    "that way. Then, Speaker tells his story in a conversation with a second worker "
    "(Listener). The emotion label and situation of Speaker are invisible to Listener. "
    "Listener should recognize and acknowledge others' feelings in a conversation as "
    "much as possible. You are an empathetic conversational AI chatbot that can "
    "empathize with users. You only need to provide the next round of response of "
    "Listener. Reply with 1 sentence, maximum 15 words, natural and conversational, no emoji."
)


async def empathy_zero_shot_short(dialogue_context: str, llm) -> dict:
    response = await llm.generate(
        SYSTEM_PROMPT,
        f"Dialogue:\n{dialogue_context}",
        max_tokens=64,
        temperature=0.1,
    )

    return {
        "response": response,
        "llm_calls": 1,
    }
