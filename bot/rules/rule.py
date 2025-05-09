import random

KEYWORD = {
    "sapa": ("halo", "hello", "hai", "helohey", "hi"),
    "instrumen": ("crypto", "saham", "kripto", "bitcoin"),
    "strategi": ("strategi", "strategy", "metode"),
    "signal": ("sinyal", "signal"),
}

RULES = {
    "sapa": [
        "Halo bang! Lagi ngapain nih? Nyari-nyari cuan atau nyari masalah sama market? 😆",
        "Halooo~ Kalo ketemu aku di chart mah jangan ditembak ya! 😂",
        "Hai hai~ Siap-siap nyemplung di market hari ini? Jangan lupa pake pelampung (SL) ya! 🏊‍♂️",
    ],
    "instrumen": [
        "Waduh maap bang, aku cuma jago ngomongin forex doang nih. Kripto mah masih takut volatilitasnya 😅",
        "Ngomongin crypto? Aduh aku mah mending EUR/USD aja deh, gak berani sama si Bitcoin galak 🥶",
        "Saham kripto? Itu mah buat anak-anak muda pemberani! Aku mah mendingan main aman di forex 😎",
    ],
    "strategi": [
        "Gue pake jurus andalan: *SMC* alias 'Sok Mau Cuan'! 😆 Tapi beneran deh, ini strategi Smart Money Concept yang bikin gue jarang kena cutloss!",
        "Strategi gue simpel: Kalau market lagi baik, masuk. Lagi galak? Ngumpet dulu! 😂 Pake SMC biar gak asal entry~",
        "Jurus rahasia gue: Analisa dulu, trading belakangan. Kalo udah profit, cabut duluan! Gitu aja kok repot 😝",
    ],
    "signal": [
        "Mau sinyal? Boleh banget! Tapi jangan lupa, aku cuma kasih saran ya. Yang mutusin tetap elo! 😘 Ketik /menu yuk~",
        "Sinyal mah ada banyak, tapi jangan asal nyemplung! Nanti kena cutloss malah nyesel 😆 Cek /menu deh!",
        "Waduh, sinyal hari ini lagi pada manis-manis nih~ Tapi tunggu dulu, aku kasih tau via /menu ya biar gak salah entry!",
    ],
}


def get_response(user_msg):
    user_msg = user_msg.lower()
    for word in user_msg.split():
        for group, keyword in KEYWORD.items():
            if word in keyword:
                return random.choice(RULES[group])
    return "Waduh, gak ngerti nih maksud lo apa, aku cuma jago bahas forex doang sih 😅"
