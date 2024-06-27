import os
import re
import pickle
from collections import Counter
import numpy as np
from structures import trie, graph

def highlight_term(text, term):
    return text.replace(term, f"\033[1;32;40m{term}\033[0m")

def get_sentence_contexts(text, term, context_sentences=2):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    term_sentence_indices = [i for i, sentence in enumerate(sentences) if term in sentence]
    
    contexts = []
    for term_sentence_idx in term_sentence_indices:
        start_idx = max(0, term_sentence_idx - context_sentences)
        end_idx = min(len(sentences), term_sentence_idx + context_sentences + 1)
        context = " ".join(sentences[start_idx:end_idx])
        contexts.append(context)
    
    return contexts

def search_and_display_results(results, term, start_rank=1, results_per_page=10, context_sentences=2):
    if not results:
        print("No results found")
        return

    rank = start_rank

    print(f"Total results found: {len(results)}\n")

    seen_pages = set()
    for result in results:
        page_index = result.page
        if page_index in seen_pages:
            continue
        seen_pages.add(page_index)

        print("______________________________")
        print(f"Rank: {rank}")
        print(f"Page: {page_index}")
        print(f"Occurrences: {result.occurrences}")

        page_text = load_page_text(page_index)
        if page_text:
            contexts = get_sentence_contexts(page_text, term, context_sentences)
            unique_contexts = list(dict.fromkeys(contexts))  # Preserve order and remove duplicates
            for context in unique_contexts:
                print(f"Context:\n{context}")
                print()

        rank += 1

        if rank - start_rank >= results_per_page:
            print("------------------------------")
            print(f"Type 'n' for next page or any other key to exit.")
            print("------------------------------")
            user_input = input().strip().lower()
            if user_input != 'n':
                return

def and_search(query, trie_structure, context_sentences=2):
    if len(query) < 2:
        print("AND operation requires at least two search terms.")
        return

    term_results = [set(trie_structure.search(term)) for term in query]
    common_pages = set.intersection(*term_results)
    
    if not common_pages:
        print("No results found for the AND operation.")
    else:
        combined_results = []
        for page in common_pages:
            for term in query:
                combined_results.extend(trie_structure.search(term))
        search_and_display_results(combined_results, query[0], context_sentences=context_sentences)

def or_search(query, trie_structure, context_sentences=2):
    results = {}
    for term in query:
        term_results = trie_structure.search(term)
        if term_results:
            for result in term_results:
                if result.page in results:
                    results[result.page].append(result)
                else:
                    results[result.page] = [result]

    combined_results = [item for sublist in results.values() for item in sublist]
    search_and_display_results(combined_results, query[0], context_sentences=context_sentences)

def not_search(query, trie_structure, context_sentences=2):
    if len(query) != 1:
        print("NOT operation requires exactly one search term.")
        return
    
    not_results = []
    for term in trie_structure.root.children:
        if term != query[0]:
            term_results = trie_structure.search(term)
            if term_results:
                not_results.extend(term_results)

    if not_results:
        search_and_display_results(not_results, query[0], context_sentences=context_sentences)
    else:
        print(f"No results found for the NOT operation for term '{query[0]}'.")

def single_word_search(query, trie_structure, context_sentences=2):
    search_results = trie_structure.search(query[0])
    search_and_display_results(search_results, query[0], context_sentences=context_sentences)

def page_rank(graph, damping_factor=0.85, max_iterations=100, tol=1.0e-6):
    n = graph.vertex_count()
    ranks = np.full(n, 1.0 / n)
    new_ranks = np.zeros(n)
    
    vertices = list(graph.vertices())
    vertex_indices = {v: i for i, v in enumerate(vertices)}
    
    for _ in range(max_iterations):
        for v in vertices:
            rank_sum = 0
            for e in graph.incident_edges(v, outgoing=False):
                u = e.opposite(v)
                rank_sum += ranks[vertex_indices[u]] / graph.degree(u, incoming=False)
            
            new_ranks[vertex_indices[v]] = (1 - damping_factor) / n + damping_factor * rank_sum
        
        if np.linalg.norm(new_ranks - ranks) < tol:
            break
        ranks, new_ranks = new_ranks, ranks
    
    return ranks

def count_word_occurrences(pages_text, search_term):
    term_counts = []
    for text in pages_text:
        words = text.lower().split()
        term_count = Counter(words)[search_term.lower()]
        term_counts.append(term_count)
    return term_counts

def rank_pages(page_ranks, term_counts, alpha=0.5):
    combined_scores = [(alpha * page_ranks[i] + (1 - alpha) * term_counts[i], i) for i in range(len(page_ranks))]
    combined_scores.sort(reverse=True, key=lambda x: x[0])
    return combined_scores

def turn_pages_to_graph(txt_directory):
    pages_text = []
    pages_links = []  # Replace with actual link extraction logic if needed

    for filename in sorted(os.listdir(txt_directory)):
        if filename.endswith(".txt"):
            with open(os.path.join(txt_directory, filename), "r", encoding="utf-8") as file:
                text = file.read()
                pages_text.append(text)
                # Dummy links; replace with actual logic if available
                pages_links.append([])

    g = graph.Graph()
    vertices = []

    for i, text in enumerate(pages_text):
        v = g.insert_vertex(x={'text': text, 'index': i})
        vertices.append(v)

    # Assuming links between consecutive pages; replace with actual link extraction logic if available
    for i in range(len(vertices) - 1):
        g.insert_edge(vertices[i], vertices[i + 1])

    return g, vertices, pages_text

def load_page_text(page_number):
    file_path = f"./txts/page_{page_number}.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    return ""

def load_and_parse(txt_directory):
    try:
        trie_structure = load_words_into_trie(txt_directory)
        return trie_structure
    except Exception as e:
        print(f"An error occurred while loading and parsing the TXT files: {e}")
        return None

def load_words_into_trie(txt_directory):
    try:
        if not os.path.exists(txt_directory):
            os.makedirs(txt_directory)
        
        trie_structure = trie.Trie()
        total_words_inserted = 0
        pages_processed = 0

        for filename in sorted(os.listdir(txt_directory)):
            if filename.endswith(".txt"):
                page_number = int(filename.split('_')[1].split('.')[0])
                with open(os.path.join(txt_directory, filename), "r", encoding="utf-8") as file:
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

def search_query(query, trie_structure, context_sentences=2, txt_directory=None):
    query = query.split()
    print(f"Searching for term \033[1;32;40m{' '.join(query)}\033[0m...")

    if txt_directory:
        g, vertices, pages_text = turn_pages_to_graph(txt_directory)
        page_ranks = page_rank(g)
        term_counts = count_word_occurrences(pages_text, query[0])
        combined_scores = rank_pages(page_ranks, term_counts)
        search_and_display_results(combined_scores, query[0], context_sentences=context_sentences)
    else:
        if "AND" in query:
            query.remove("AND")
            and_search(query, trie_structure, context_sentences=context_sentences)

        elif "OR" in query:
            query.remove("OR")
            or_search(query, trie_structure, context_sentences=context_sentences)

        elif "NOT" in query:
            query.remove("NOT")
            not_search(query, trie_structure, context_sentences=context_sentences)

        else:
            if len(query) == 1:
                single_word_search(query, trie_structure, context_sentences=context_sentences)
            else:
                print("Invalid query format. Please enter a single word or a valid logical operation.")
