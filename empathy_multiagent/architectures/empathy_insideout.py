# architectures/empathy_insideout.py
# InsideOut architecture (Setting 1, emotion_reconcile-langchain_dev, ACL 2024)
#
# Pipeline (6 LLM-вызовов):
#   Dialogue
#     → [1-4. Emotion Agents] параллельно: happiness, anger, sadness, fear
#            каждый оценивает эмоцию первого говорящего со своей перспективы
#     → [5. Aggregator] объединяет 4 оценки в итоговый тег
#     → [6. Responder] генерирует эмпатичный ответ с учётом эмоции
#     → Response

import asyncio

# ── Emotion agent role descriptions (prompts/emotions/v1/) ────────────────────

_EMOTION_DESCRIPTIONS = {
    "happiness": 'You are representing "happiness" emotion, meaning that your judgments and decisions should be dictated by this emotion.',
    "anger":     'You are representing "anger" emotion, meaning that your judgments and decisions should be dictated by this emotion.',
    "sadness":   'You are representing "sadness" emotion, meaning that your judgments and decisions should be dictated by this emotion.',
    "fear":      'You are representing "fear" emotion, meaning that your judgments and decisions should be dictated by this emotion.',
}

# ── Step 1: each agent assesses speaker emotion (setting_1/v2/step_1_emotions.txt) ──

_STEP1_SYSTEM_TEMPLATE = """{agent_description}

You are given a dialogue. You need to assess the emotion of the first speaker, estimate your confidence in this assessment, and provide reasoning for your answer.
Your response should be in the following format:
`Emotion;Confidence;Reasoning`
For example: `Sad;0.5;The speaker implied that they are unsatisfied with their life.`
The possible emotions are guilty, angry, sad, surprise, anxious, jealous, fear, disgusted, sentimental, happy, hopeful, grateful, caring, embarrassed, trusting, proud, and confident.
Confidence estimation is a float number from 0 to 1.
Reasoning is your thoughts behind your answer. Keep your reasoning concise."""

# ── Step 2: aggregator combines 4 assessments (setting_1/v2/step_2_aggregation.txt) ─

_STEP2_SYSTEM_TEMPLATE = """You are the final judge over the assessment of the emotional condition of the speaker of the dialog above.
Look at hints other agents provide you and construct a final answer.
Happiness says:
{happiness}
Anger says:
{anger}
Sadness says:
{sadness}
Fear says:
{fear}

The answer should be in the following format:
`Emotion;Confidence;Reasoning`
For example: `Sad; 0.5; The speaker implied that he is unsatisfied with his life.`
The emotions can be guilty, angry, sad, surprise, anxious, jealous, fear, disgusted, sentimental, happy, hopeful, grateful, caring, embarrassed, trusting, proud, confident.
Confidence estimation is a float number from 0 to 1.
Reasoning is the thoughts behind your answer. Keep your reasoning concise."""

# ── Step 3: generate empathetic response (setting_1/v2/step_3_final_answer.txt) ──

_STEP3_USER_TEMPLATE = """Current state of a dialog:
{dialog}
Use the following information about the emotional state of the first speaker:
{emotional_state_info}
Answer as if you are Speaker 2. Use the information about the emotional state to craft a response aimed at improving the emotional state of Speaker 1, if necessary.
But it is also important not to overreact and keep your answer concise."""


# ── Helper ────────────────────────────────────────────────────────────────────

def _parse_semicolon(raw: str) -> tuple:
    cleaned = raw.strip().strip("`")
    parts = [p.strip() for p in cleaned.split(";", 2)]
    while len(parts) < 3:
        parts.append("")
    return parts[0], parts[1], parts[2]


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def empathy_insideout(dialogue_context: str, llm, retriever=None) -> dict:
    # ── Step 1: 4 emotion agents in parallel ─────────────────────────────────
    async def _run_agent(description: str) -> str:
        system = _STEP1_SYSTEM_TEMPLATE.format(agent_description=description)
        return await llm.generate(
            system,
            f"The dialogue is as follows:\n{dialogue_context}",
            temperature=0.0,
            max_tokens=150,
        )

    agent_outputs = await asyncio.gather(*[
        _run_agent(desc) for desc in _EMOTION_DESCRIPTIONS.values()
    ])
    assessments = dict(zip(_EMOTION_DESCRIPTIONS.keys(), agent_outputs))

    # ── Step 2: Aggregator ────────────────────────────────────────────────────
    agg_system = _STEP2_SYSTEM_TEMPLATE.format(**assessments)
    agg_raw = await llm.generate(
        agg_system,
        dialogue_context,
        temperature=0.0,
        max_tokens=150,
    )
    tag, confidence, reasoning = _parse_semicolon(agg_raw)

    # Format matches original lambda: names = ["tag: ", "confidence: ", "additional information: "]
    emotional_state_info = (
        f"tag: {tag}\n"
        f"confidence: {confidence}\n"
        f"additional information: {reasoning}"
    )

    # ── Step 3: Final response ────────────────────────────────────────────────
    user_msg = _STEP3_USER_TEMPLATE.format(
        dialog=dialogue_context,
        emotional_state_info=emotional_state_info,
    )
    response = await llm.generate(
        "You are a compassionate dialogue assistant.",
        user_msg,
        temperature=0.7,
        max_tokens=200,
    )

    return {
        "response": response,
        "emotion": {"emotion": tag, "confidence": confidence, "reasoning": reasoning},
        "emotion_assessments": assessments,
        "llm_calls": 6,
    }