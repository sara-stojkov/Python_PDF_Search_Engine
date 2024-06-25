import os
from structures import trie

def highlight_term(text, term):
    return text.replace(term, f"\033[1;32;40m{term}\033[0m")

def get_sentence_context(text, position, term, context_sentences=2):
    import re
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    cumulative_length = 0
    term_sentence_idx = None
    for i, sentence in enumerate(sentences):
        cumulative_length += len(sentence) + 1  # +1 for the removed space
        if cumulative_length > position:
            term_sentence_idx = i
            break
    if term_sentence_idx is None:
        return "", ""
    term_sentence = sentences[term_sentence_idx].strip()
    highlighted_sentence = highlight_term(term_sentence, term)

    start_idx = max(0, term_sentence_idx - context_sentences)
    end_idx = min(len(sentences), term_sentence_idx + 1 + context_sentences)

    context_sentences_list = sentences[start_idx:end_idx]
    context_text = ". ".join(context_sentences_list).strip()

    return highlighted_sentence, context_text

def search_and_display_results(results, term, start_rank=1, results_per_page=10, context_sentences=2):
    if not results:
        print("No results found")
        return

    sorted_results = sorted(results, key=lambda x: x.occurrences, reverse=True)
    rank = start_rank
    seen_pages = set()

    print(f"Total results found: {len(sorted_results)}\n")

    for result in sorted_results:
        if result.page in seen_pages:
            continue

        seen_pages.add(result.page)

        print(f"Rank: {rank}")
        print(f"Page: {result.page}")
        print(f"Occurrences: {result.occurrences}")

        page_text = load_page_text(result.page)

        if page_text:
            for position in result.positions:
                highlighted_sentence, context = get_sentence_context(page_text, position, term, context_sentences)

                if highlighted_sentence and context:
                    print(f"Context: {context}")
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

    term_results = []
    for term in query:
        results = trie_structure.search(term)
        if results:
            term_results.append(set(results))
        else:
            term_results.append(set())

    common_pages = set.intersection(*term_results)
    if not common_pages:
        print("No results found for the AND operation.")
    else:
        combined_results = []
        for page in common_pages:
            combined_results.extend(trie_structure.search(query[0]))
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

    combined_results = []
    for page_results in results.values():
        combined_results.extend(page_results)
    
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

def search_query(query, trie_structure, context_sentences=2):
    query = query.split()
    print(f"Searching for term \033[1;32;40m{' '.join(query)}\033[0m...")

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

def load_page_text(page_number):
    file_path = f"./txts/page_{page_number}.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    return ""
