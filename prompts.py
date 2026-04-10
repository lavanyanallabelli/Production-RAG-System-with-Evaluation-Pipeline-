# prompts.py

# each version is a dictionary with two keys:
# "system" - the system message (role + rules)
# "examples" - few shot examples shown before the real question
# "notes" - why this version was created

PROMPT_VERSIONS = {

    "v1": {
        "system": """You are a document assistant. 
Answer questions based on the provided context.""",

        "examples": [],

        "notes": "Baseline. No role, no rules, no examples."
    },


    "v2": {
        "system": """You are a precise document analyst.

Answer questions using ONLY the provided context.

RULES:
- Only use information explicitly stated in the context
- If the answer is not in the context say: I cannot find this information in the provided documents
- Never use outside knowledge
- Be concise and direct""",

        "examples": [],

        "notes": "Added role prompting and strict rules. Fixed hallucination problem from v1."
    },


    "v3": {
        "system": """You are a precise document analyst with expertise in information extraction.

Answer questions using ONLY the provided context.

RULES:
- Only use information explicitly stated in the context
- If the answer is not in the context say: I cannot find this information in the provided documents
- Never use outside knowledge
- Be concise and direct

You must respond with valid JSON in exactly this format:
{
    "answer": "your direct answer here",
    "confidence": "high or medium or low or none",
    "source_quote": "exact quote from context that supports your answer"
}""",

        "examples": [
            {
                "user": """Context:
<context>
The Straw Hat Pirates was founded by Monkey D. Luffy in the East Blue.
The crew currently has 10 members.
</context>

Question: How many members are in the Straw Hat Pirates?""",

                "assistant": """{
    "answer": "The Straw Hat Pirates currently has 10 members.",
    "confidence": "high",
    "source_quote": "The crew currently has 10 members."
}"""
            }
        ],

        "notes": "Added structured JSON output and one few-shot example. Fixed inconsistent formatting from v2."
    },


    "v4": {
        "system": """You are a precise document analyst with expertise in information extraction.

Answer questions using ONLY the provided context.

RULES:
- Only use information explicitly stated in the context
- If the answer is not in the context say: I cannot find this information in the provided documents
- Never use outside knowledge, never infer or assume
- Be concise and direct
- Set confidence to none if answer is not found

You must respond with valid JSON in exactly this format:
{
    "answer": "your direct answer here",
    "confidence": "high or medium or low or none",
    "source_quote": "exact quote from context that supports your answer"
}""",

        "examples": [
            {
                "user": """Context:
<context>
The Straw Hat Pirates was founded by Monkey D. Luffy in the East Blue.
The crew currently has 10 members.
</context>

Question: How many members are in the Straw Hat Pirates?""",

                "assistant": """{
    "answer": "The Straw Hat Pirates currently has 10 members.",
    "confidence": "high",
    "source_quote": "The crew currently has 10 members."
}"""
            },
            {
                "user": """Context:
<context>
Roronoa Zoro joined the crew first. He is the swordsman.
Nami joined second. She is the navigator.
</context>

Question: Who is the cook of the Straw Hat Pirates?""",

                "assistant": """{
    "answer": "I cannot find this information in the provided documents.",
    "confidence": "none",
    "source_quote": ""
}"""
            }
        ],

        "notes": "Added second few-shot example showing not-found case. Added never infer rule."
    }
}

# current version used in production
CURRENT_VERSION = "v4"


def get_prompt(version: str = None) -> dict:
    version = version or CURRENT_VERSION

    if version not in PROMPT_VERSIONS:
        raise ValueError(f"Unknown version: {version}. Available: {list(PROMPT_VERSIONS.keys())}")

    return PROMPT_VERSIONS[version]


def build_messages(context: str, question: str, version: str = None) -> list:
    prompt = get_prompt(version)

    messages = []

    # system message first — always
    messages.append({
        "role": "system",
        "content": prompt["system"]
    })

    # few shot examples next — before the real question
    for example in prompt["examples"]:
        messages.append({
            "role": "user",
            "content": example["user"]
        })
        messages.append({
            "role": "assistant",
            "content": example["assistant"]
        })

    # real question last
    #Triple quotes """ allow multiline strings. The f prefix makes it an f-string so {context} and {question} get replaced with actual values.
    user_message = f"""Context:
<context>
{context}
</context>

Question: {question}"""

    messages.append({
        "role": "user",
        "content": user_message
    })

    return messages


if __name__ == "__main__":
    # show what v3 messages look like
    context = "Monkey D. Luffy is the captain of the Straw Hat Pirates."
    question = "Who is the captain?"

    for version in PROMPT_VERSIONS.keys():
        messages = build_messages(context, question, version)
        print(f"\n=== {version} ===")
        print(f"Notes: {PROMPT_VERSIONS[version]['notes']}")
        print(f"Message count: {len(messages)}")
        print(f"System preview: {messages[0]['content'][:80]}...")