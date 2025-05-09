import random

RULES = {
    "halo": [
        "*Halo juga, trader sejati!* 🚀\nGimana market hari ini? Masih bersahabat atau lagi ngajak ribut? 😂 Tenang, aku siap bantu kamu mantau pergerakan harga dan cari peluang terbaik buat dapet cuan! 💸",
        "*Hai, apa kabar?* 📈\nSemoga hari ini banyak setup valid yang muncul ya! Jangan FOMO, analisa dulu... entry belakangan. Yang penting sabar, bukan barbar! 😎",
        "Halo, pasti kamu lagi nyari signal trading kan? wkwkwk 😂 Tenang, kamu datang ke tempat yang tepat! Aku siap kasih info dan sinyal yang bisa bantu kamu ambil keputusan dengan lebih percaya diri. 💡",
        "*Halo, sang pemburu pips!* 💸\nMau scalping atau swing, semua butuh perencanaan. Aku bantu pantau chart, indikator, dan data yang kamu perluin biar entry gak asal-asalan. 🤓",
        "*Hai, balik lagi ya?* ⏳\nMarket forex emang dinamis banget. Tapi selama kamu pake analisa dan risk management, hasilnya pasti lebih terarah. Jangan sampe MC ya, nanti nangis! 😭",
    ],
    "crypto": ["Maaf, untuk saat ini saya hanya bisa analisa market forex saja 😅,"],
    "strategi": [
        "Saya di program menggunakan strategi *Smart Money Concept (SMC).* 📊💡 Strategi ini fokus untuk mengikuti aliran besar di pasar, jadi lebih mengutamakan pergerakan besar yang dipengaruhi oleh institutional traders. Dengan kombinasi indikator teknikal 📈, saya bantu pantau market dan cari peluang entry yang lebih terarah. Keren, kan? 😎",
    ],
    "akurat": [""],
}


def get_response(user_msg):
    user_msg = user_msg.lower()
    for keyword, responses in RULES.items():
        if keyword in user_msg:
            return random.choice(responses)
    return "Maaf, saya tidak tahu maksudmu. 😕 Saya punya batasan dalam berbicara, jadi kalau kamu punya pertanyaan tentang analisa market, signal trading, atau strategi forex, silakan tanyakan! 📈💬 Kalau bukan itu, mungkin saya gak bisa bantu banyak deh. 😅"
