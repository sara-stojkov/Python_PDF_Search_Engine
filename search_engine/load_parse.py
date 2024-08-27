import fitz
import pickle
import os
import re
from structures.trie import Trie, NodeInfo
from structures.graph import Graph

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_DIR = os.path.join(BASE_DIR, "txts")
TRIE_PICKLE = os.path.join(BASE_DIR, "parsed_trie.pickle")
GRAPH_PICKLE = os.path.join(BASE_DIR, "reference_graph.pickle")
PAGE_OFFSET = 22

def load_and_parse(file_path):
    try:
        if os.path.exists(TRIE_PICKLE) and os.path.exists(GRAPH_PICKLE):
            trie_structure, graph, vertices = load_pickle_files()
            print("Raniji trie i graf učitani iz pickle fajlova.")
            return trie_structure, graph, vertices

        pdf_document = fitz.open(file_path)
        pdf_page_count = pdf_document.page_count
        load_pages(pdf_document)
        pdf_document.close()

        print("PDF uspešno učitan u txt fajlove.")
        trie_structure = load_words_into_trie()
        graph, vertices = build_reference_graph(pdf_page_count)
        save_graph(graph, GRAPH_PICKLE)
        print("Graf referenci uspešno napravljen i sačuvan u pickle fajlu.")
        return trie_structure, graph, vertices
    except Exception as e:
        print(f"Došlo je do greške pri parsiranju PDF-a: {e}")
        return None, None, None

def load_pickle_files():
    with open(TRIE_PICKLE, "rb") as trie_file, open(GRAPH_PICKLE, "rb") as graph_file:
        trie_structure = pickle.load(trie_file)
        graph = pickle.load(graph_file)
        vertices = {v.data: v for v in graph.vertices()}
    return trie_structure, graph, vertices

def load_pages(pdf_document):
    try:
        os.makedirs(TXT_DIR, exist_ok=True)
        for i in range(pdf_document.page_count):
            page = pdf_document.load_page(i)
            page_text = page.get_text()
            with open(os.path.join(TXT_DIR, f"page_{i+1}.txt"), "w", encoding="utf-8") as file:
                file.write(page_text)
    except Exception as e:
        print(f"Došlo je do greške pri učitavanju PDF-a: {e}")

def load_words_into_trie():
    try:
        os.makedirs(TXT_DIR, exist_ok=True)
        trie_structure = Trie()
        total_words_inserted = 0
        pages_processed = 0

        for filename in sorted(os.listdir(TXT_DIR)):
            page_number = int(filename.split('_')[1].split('.')[0])
            with open(os.path.join(TXT_DIR, filename), "r", encoding="utf-8") as file:
                page_text = file.read()
                words = re.findall(r'\b\w+\b', page_text.lower())
                word_positions = {}

                for position, word in enumerate(words):
                    word_positions.setdefault(word, []).append(position)
                
                for word, positions in word_positions.items():
                    occurrences = len(positions)
                    node_info = NodeInfo(page_number, positions, occurrences)
                    trie_structure.insert(word, node_info)
                    total_words_inserted += occurrences
            
            pages_processed += 1

        with open(TRIE_PICKLE, "wb") as file:
            pickle.dump(trie_structure, file)

        print(f"Ukupno ubačenih reči u trie: {total_words_inserted}")
        print(f"Ukupno obrađenih stranica: {pages_processed}")
        
        return trie_structure
    except Exception as e:
        print(f"Došlo je do greške pri učitavanju reči u trie: {e}")
        return None

def load_page_text(page_number):
    file_path = os.path.join(TXT_DIR, f"page_{page_number}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    return ""

def build_reference_graph(pdf_page_count, offset=PAGE_OFFSET):
    graph = Graph(directed=True)
    reference_pattern = re.compile(r'\b(?:from|on|see|as described on|as discussed on|refer to|in)\s+(?:page|pg|p)\s+(\d{1,3})\b', re.IGNORECASE)
    
    vertices = {}
    for page in range(1, pdf_page_count + 1):
        page_text = load_page_text(page)
        if page_text:
            page_vertex = vertices.get(page, graph.insert_vertex(page))
            vertices[page] = page_vertex

            matches = reference_pattern.findall(page_text)
            for match in matches:
                referenced_page = int(match)
                actual_referenced_page = referenced_page + offset  # Adjust for offset
                if 1 <= actual_referenced_page <= pdf_page_count:
                    referenced_page_vertex = vertices.get(actual_referenced_page, graph.insert_vertex(actual_referenced_page))
                    vertices[actual_referenced_page] = referenced_page_vertex
                    graph.insert_edge(page_vertex, referenced_page_vertex)

    return graph, vertices

def save_graph(graph, filename):
    with open(filename, "wb") as file:
        pickle.dump(graph, file)
