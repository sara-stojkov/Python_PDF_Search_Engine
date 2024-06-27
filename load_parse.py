import fitz  
import pickle
import os
import re
from structures.trie import Trie, NodeInfo, TrieNode
from structures.graph import Graph

def load_and_parse(file_path):
    try:
        load_pages(file_path)
        print("PDF successfully converted to text files.")
        trie_structure = load_words_into_trie()
        graph, vertices = build_reference_graph()
        save_graph(graph, "reference_graph.pickle")
        print("Reference graph successfully built and saved.")
        return trie_structure, graph, vertices
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
        
        trie_structure = Trie()
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
                    node_info = NodeInfo(page_number, positions, occurrences)
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

def load_page_text(page_number):
    file_path = f"./txts/page_{page_number}.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    return ""

def build_reference_graph(offset=22):
    graph = Graph(directed=True)
    reference_pattern = re.compile(r'\b(?:from|on|see|as described on|as discussed on|refer to|in)\s+(?:page|pg|p)\s+(\d{1,3})\b', re.IGNORECASE)
    
    vertices = {}
    for page in range(1, 771):
        page_text = load_page_text(page)
        if page_text:
            page_vertex = vertices.get(page)
            if not page_vertex:
                page_vertex = graph.insert_vertex(page)
                vertices[page] = page_vertex

            matches = reference_pattern.findall(page_text)
            for match in matches:
                referenced_page = int(match)
                actual_referenced_page = referenced_page + offset  # Adjust for offset
                if 1 <= actual_referenced_page <= 770:
                    referenced_page_vertex = vertices.get(actual_referenced_page)
                    if not referenced_page_vertex:
                        referenced_page_vertex = graph.insert_vertex(actual_referenced_page)
                        vertices[actual_referenced_page] = referenced_page_vertex
                    graph.insert_edge(page_vertex, referenced_page_vertex)

    return graph, vertices

def save_graph(graph, filename):
    with open(filename, "wb") as file:
        pickle.dump(graph, file)