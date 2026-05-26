# architectures/chain_of_empathy.py
# Архитектура: Chain-of-Empathy (CoE) — CBT variant
# Based on Lee et al. (2024) arXiv:2311.04915
# Dialogue → [CoE Reasoning → Response] → Response
# LLM-вызовов: 1

SYSTEM_PROMPT = (
    "This is an empathetic dialogue task: The first worker (Speaker) is given an "
    "emotion label and writes his own description of a situation when he has felt "
    "that way. Then, Speaker tells his story in a conversation with a second worker "
    "(Listener). The emotion label and situation of Speaker are invisible to Listener. "
    "Listener should recognize and acknowledge others' feelings in a conversation as "
    "much as possible. You are an empathetic conversational AI chatbot that can "
    "empathize with users. You only need to provide the next round of response of "
    "Listener. Reply with 1-3 sentences, natural and conversational, no emoji."
)

COE_CBT_PROMPT = (
    "Before responding, reason step by step using Cognitive Behavioral Therapy (CBT) principles:\n"
    "1. Identify the speaker's emotional state and the situation triggering it.\n"
    "2. Identify any cognitive distortions or unhelpful thought patterns the speaker may have.\n"
    "3. Consider what a balanced, realistic reframe of their situation might look like.\n"
    "4. Based on this reasoning, craft a response that validates their feelings and "
    "gently offers a constructive perspective.\n\n"
    "Format your output as:\n"
    "Reasoning: <your CBT reasoning>\n"
    "Response: <your empathetic response, 1-3 sentences>\n"
)


async def chain_of_empathy(dialogue_context: str, llm) -> dict:
    raw = await llm.generate(
        SYSTEM_PROMPT,
        f"{COE_CBT_PROMPT}\nDialogue:\n{dialogue_context}",
        max_tokens=512,
        temperature=0.1,
    )

    # Извлекаем только финальный ответ после "Response:"
    if "Response:" in raw:
        response = raw.split("Response:")[-1].strip()
    else:
        response = raw.strip()

    return {
        "response": response,
        "llm_calls": 1,
    }