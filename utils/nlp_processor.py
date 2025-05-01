import itertools
from spellchecker import SpellChecker

# --- 1. Define confusion groups based on FSL similarity ---
confusion_groups = {
    'A': ['E', 'S', 'M', 'N'],
    'M': ['A', 'N', 'S', 'E'],
    'N': ['A', 'M', 'S', 'E'],
    'S': ['A', 'M', 'N', 'E'],
    'E': ['A', 'M', 'N', 'S'],
    'G': ['H', 'I'],
    'H': ['G', 'I'],
    'I': ['J'],
    'J': ['I'],
    'K': ['V', 'P', 'Q'],
    'P': ['K', 'V'],
    'V': ['K', 'P'],
    'Q': ['K'],
    'U': ['R'],
    'R': ['U'],
    'Y': ['L'],
    'L': ['Y']
}

# --- 2. Tagalog Dictionary ---
# Expanded common Tagalog words
dictionary = {
    "kumusta", "ka", "ako", "ikaw", "salamat", "mahal", "bakit", "saan", "kita",
    "oo", "hindi", "maganda", "pangit", "gutom", "pagkain", "inom", "sakit",
    "bahay", "trabaho", "araw", "gabi", "umaga", "hapon", "tulog", "lakad", "bili",
    "kape", "tubig", "bukas", "ngayon", "kanina", "mamaya"
}

# Fallback spellchecker for unknown words (Tagalog tuned manually)
spell = SpellChecker(language=None)
spell.word_frequency.load_words(dictionary)

# --- 3. Utility Functions ---
def compress_repeats(letters):
    """Compress consecutive duplicate letters."""
    if not letters:
        return []
    compressed = [letters[0]]
    for letter in letters[1:]:
        if letter != compressed[-1]:
            compressed.append(letter)
    return compressed

def generate_alternatives(word):
    """Generate all possible letter combinations based on confusion groups."""
    options = []
    for letter in word:
        if letter.upper() in confusion_groups:
            alternatives = [letter] + [c.lower() for c in confusion_groups[letter.upper()]]
        else:
            alternatives = [letter]
        options.append(alternatives)
    return options

def find_best_match(raw_word):
    """Find the best valid Tagalog word using confusion groups."""
    if raw_word in dictionary:
        return raw_word

    # Generate possible variants
    alternatives = generate_alternatives(raw_word)
    candidates = map(''.join, itertools.product(*alternatives))

    # Try candidates
    for candidate in candidates:
        if candidate in dictionary:
            return candidate

    # Spellcheck fallback
    suggestion = spell.correction(raw_word)
    if suggestion:
        return suggestion

    return raw_word  # Fallback if no match

def greedy_split(raw_text):
    """Split text into words based on the dictionary, greedy matching."""
    words = []
    i = 0
    while i < len(raw_text):
        matched = False
        # Try the longest possible word first
        for j in range(len(raw_text), i, -1):
            segment = raw_text[i:j]
            if segment in dictionary:
                words.append(segment)
                i = j
                matched = True
                break
        if not matched:
            # No word matched; move one letter
            words.append(raw_text[i])
            i += 1
    return words

# --- 4. Main Sentence Generator ---

def generate_tagalog_sentence(letters):
    """Main function to process recognized letters into a proper Tagalog sentence."""
    if not letters:
        return ""

    # Step 0: Clean and lowercase
    letters = [l.lower() for l in letters if l is not None]

    # Step 1: Compress duplicates
    letters = compress_repeats(letters)

    # Step 2: Join into raw string
    raw_text = ''.join(letters)

    # Step 3: Greedy split into words
    rough_words = greedy_split(raw_text)

    # Step 4: Confusion correction + spell correction
    corrected_words = []
    for word in rough_words:
        corrected_word = find_best_match(word)
        corrected_words.append(corrected_word)

    # Step 5: Assemble final sentence
    sentence = " ".join(corrected_words)
    sentence = sentence.capitalize()

    return sentence
