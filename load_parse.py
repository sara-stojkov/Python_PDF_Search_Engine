import fitz  
import pickle
import os
from structures import trie

def load_and_parse(file_path):
    try:
        load_pages(file_path)
        print("PDF successfully converted to text files.")
        trie_structure = load_words_into_trie()
        return trie_structure
    except Exception as e:
        print(f"An error occurred while loading and parsing the PDF: {e}")
        return None

def load_pages(file_path):
    try:
        pdf_document = fitz.open(file_path)
        number_of_pages = pdf_document.page_count
        for i in range(number_of_pages):
            page = pdf_document.load_page(i)
            page_text = page.get_text()
            with open(f"txts/page_{i+1}.txt", "w", encoding="utf-8") as file:
                file.write(page_text)
        pdf_document.close()
    except Exception as e:
        print(f"An error occurred while loading pages from the PDF: {e}")

def load_words_into_trie():
    try:
        directory = "txts"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        trie_structure = trie.Trie()
        total_words_inserted = 0
        pages_processed = 0

        for filename in sorted(os.listdir("txts")):
            page_number = int(filename.split('_')[1].split('.')[0])
            with open(os.path.join("txts", filename), "r", encoding="utf-8") as file:
                page_text = file.read()
                words = page_text.split()
                words = [word.strip(".,!?").lower() for word in words]
                word_positions = {}
                for position, word in enumerate(words):
                    if word not in word_positions:
                        word_positions[word] = []
                    word_positions[word].append(position)
                for word, positions in word_positions.items():
                    occurrences = len(positions)
                    node_info = trie.NodeInfo(page_number, positions, occurrences)
                    trie_structure.insert(word, node_info)
                    total_words_inserted += occurrences
            pages_processed += 1

        with open("parsed_trie.pickle", "wb") as file:
            pickle.dump(trie_structure, file)

        print(f"Total words inserted: {total_words_inserted}")
        print(f"Total pages processed: {pages_processed}")
        
        return trie_structure
    except Exception as e:
        print(f"An error occurred while loading words into trie: {e}")
        return None

def page_rank_graphing():
    pass
