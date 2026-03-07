from engine import load_fonts, render_sentence, render_face, choose_best_font, filter_by_dominant_direction, dominant_direction, wrap_text, shape_text
import os


FONT_BASE_DIR = "../fonts/fonts_data"

SELECTED_PROFILES = ["PLAIN", "OLD_DOCUMENT"]

# sentence = "𑯓𑯑𑯃𑯁𑯂𑯆, 𑯖𑯇 𑯁𑯖𑯔𑯖, 𑯜𑯂̃ 𑯖𑯐-𑯌𑯖 𑯉𑯖𑯁𑯂𑯆"
sentence = "મમ્મી, ચાલો આપણે આવતીકાલે ફિલ્મ જોવા જઇએ."

font_dir = os.path.join(FONT_BASE_DIR, 'Gujr')

fonts = load_fonts(font_dir)

font_data = choose_best_font(sentence, fonts)
font_path = font_data["path"]

sentence = "".join(
    c for c in sentence
    if (
            c in font_data["supported"]
            or unicodedata.category(c) in ("Cf", "Zs")
    )
)

# print(sentence)

sentence = filter_by_dominant_direction(sentence)

print(sentence)

# font_path = font_data["path"]

print(dominant_direction(sentence))

print(wrap_text(sentence, font_path))

print(font_path)

shape_text(sentence, font_path)