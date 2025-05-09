import random

RULES = {
    "halo": [
        "Halo juga!",
        "Hai, apa kabar?",
        "Halo, pasti kamu lagi nyari signal trading kan? wkwkwk",
    ]
}


def get_response(user_msg):
    user_msg = user_msg.lower()
    for keyword, responses in RULES.items():
        if keyword in user_msg:
            return random.choice(responses)
    return "Maaf, saya tidak tahu maksudmu. Saya memiliki batasan berbicara."
