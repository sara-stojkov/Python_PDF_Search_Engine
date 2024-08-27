import os
import re
from structures.trie import NodeInfo
from search_engine.load_parse import load_page_text, build_reference_graph

class NumberOfParametersException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def highlight_term(text, term):
    return text.replace(term, f"\033[1;32;40m{term}\033[0m")

def get_first_occurrence_context(text, term, context_sentences=2):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    for sentence in sentences:
        if term in sentence:
            return sentence
    return ""

def search_and_display_results(results, terms, start_rank=1, results_per_page=5, context_sentences=2):
    if not results:
        print("Nema rezultata za pretragu.")
        return

    rank = start_rank
    total_results = len(results)
    page_offset = 0

    while True:
        print(f"Ukupno rezultata: {total_results}\n")

        for i in range(page_offset, min(page_offset + results_per_page, total_results)):
            result = results[i]

            if isinstance(result, NodeInfo):
                page_index = result.page
                score = result.rank_score

            elif isinstance(result, tuple) and len(result) == 2:
                score, page_index = result
            else:
                print(f"Neočekivani tip rezultata: {type(result)}")
                continue

            print("______________________________")
            print(f"Rang: {rank}")
            print(f"Strana: {page_index}")
            print(f"Skor: {score}")

            page_text = load_page_text(page_index)
            if page_text:
                for term in terms:
                    context = get_first_occurrence_context(page_text, term, context_sentences)
                    if context:
                        highlighted_context = highlight_term(context, term)
                        print(f"Kontekst za '{term}':\n{highlighted_context}")
                        print()

            rank += 1
        
        print("------------------------------")
        print(f"Ukucaj 'n' za sledeću stranu, a bilo šta drugo da izađeš.")
        print("------------------------------")
        user_input = input().strip().lower()

        if user_input == 'n':
            page_offset += results_per_page
            if page_offset >= total_results:
                print("Nema više rezultata za prikaz.")
                break
        else:
            break

def search_multiple_words(query, trie_structure, graph=None, context_sentences=2):
    search_results = {}
    for term in query:
        results_from_trie = trie_structure.search(term)
        if results_from_trie:
            for result in results_from_trie:
                page_index = result.page
                if page_index in search_results:
                    search_results[page_index]['terms'].append(term)
                    search_results[page_index]['nodes'].append(result)
                else:
                    search_results[page_index] = {'terms': [term], 'nodes': [result]}
    
    if not search_results:
        print(f"Nema rezultata za '{' '.join(query)}'.")
    else:
        page_ranks = page_rank(graph, damping_factor=0.85) if graph else [0] * len(graph.vertices())
        term_counts = {term: count_word_occurrences(graph, term) for term in query} if graph else {}

        combined_results = []
        for page_index, data in search_results.items():
            terms = data['terms']
            nodes = data['nodes']
            score = sum(calculate_score(node, page_ranks, term_counts.get(term, [])) for node, term in zip(nodes, terms))

            if len(set(terms)) == len(query):
                score *= 2 

            combined_results.append((score, page_index))

        combined_results.sort(key=lambda x: x[0], reverse=True)
        search_and_display_results(combined_results, query, context_sentences=context_sentences)

def search_and(query, trie_structure, graph=None, context_sentences=2):
    term_results = [trie_structure.search(term) for term in query]
    page_count = {}
    for result_list in term_results:
        for result in result_list:
            if result.page in page_count:
                page_count[result.page] += 1
            else:
                page_count[result.page] = 1

    common_pages = {page for page, count in page_count.items() if count == len(query)}
    if not common_pages:
        print("Nema rezultata za AND operaciju.")
        return
    else:
        page_ranks = page_rank(graph)
        term_counts = {term: count_word_occurrences(graph, term) for term in query} if graph else {}
        combined_results = []
        for page in common_pages:
            combined_score = 0
            nodes = []
            for term in query:
                results = trie_structure.search(term)
                for result in results:
                    if result.page == page:
                        nodes.append(result)
                        combined_score += calculate_score(result, page_ranks=page_ranks, term_counts=term_counts)
            combined_results.append((combined_score, page))
        combined_results.sort(key=lambda x: x[0], reverse=True)
        search_and_display_results(combined_results, query, context_sentences=context_sentences)


def calculate_score(result, graph):
    page_ranks = graph.get_page_ranks() 
    score = 0.0

    for score, page_index in result:
        if page_index in page_ranks:
            score += page_ranks[page_index]  

    return score


def search_or(query, trie_structure, graph=None, context_sentences=2):
    search_results = {}
    for term in query:
        results_from_trie = trie_structure.search(term)
        if results_from_trie:
            for result in results_from_trie:
                page_index = result.page
                if page_index in search_results:
                    search_results[page_index]['terms'].append(term)
                    search_results[page_index]['nodes'].append(result)
                else:
                    search_results[page_index] = {'terms': [term], 'nodes': [result]}
    
    if not search_results:
        print(f"Nema rezultata za '{' '.join(query)}'.")
        return
    else:
        page_ranks = page_rank(graph, damping_factor=0.85) if graph else [0] * len(graph.vertices())
        term_counts = {term: count_word_occurrences(graph, term) for term in query} if graph else {}

        combined_results = []
        for page_index, data in search_results.items():
            terms = data['terms']
            nodes = data['nodes']
            score = sum(calculate_score(node, page_ranks, term_counts.get(term, [])) for node, term in zip(nodes, terms))
            
            combined_results.append((score, page_index))

        combined_results.sort(key=lambda x: x[0], reverse=True)
        search_and_display_results(combined_results, query, context_sentences=context_sentences)

def search_not(query1, query2, trie_structure, graph=None, context_sentences=2):
    if not graph:
        print("Greška. Nedostaju potrebni podaci.")
        return
    term_results1 = [trie_structure.search(term) for term in query1]
    term_results2 = [trie_structure.search(term) for term in query2]
    page_count1 = {}
    for result_list in term_results1:
        for result in result_list:
            if result.page in page_count1:
                page_count1[result.page] += 1
            else:
                page_count1[result.page] = 1 
    common_pages = {page for page, count in page_count1.items() if count == len(query1)} # ove treba izbaciti
    if not common_pages:
        search_and_display_results(term_results1, query1, context_sentences=context_sentences)
    else:
        page_ranks = page_rank(graph)
        term_counts = {term: count_word_occurrences(graph, term) for term in query1} if graph else {}
        combined_results = []
        starting_pages = result
        for page in common_pages:
            if page in starting_pages:
                starting_pages.remove(page)
            combined_score = 0
            nodes = []
            for term in query1:
                results = trie_structure.search(term)
                for result in results:
                    if result.page == page:
                        nodes.append(result)
                        combined_score += calculate_score(result, page_ranks=page_ranks, term_counts=term_counts)
            combined_results.append((combined_score, page))
        combined_results.sort(key=lambda x: x[0], reverse=True)
        search_and_display_results(combined_results, query1, context_sentences=context_sentences)

def suggest_alternative_queries(query, trie_structure, max_suggestions=5):
    suggestions = trie_structure.get_words_from_prefix(query)
    if not suggestions:
        return []
    def calculate_rank(query_term):
        search_results = trie_structure.search(query_term)
        if not search_results:
            return 0
        return sum(len(results) for results in search_results)
    sorted_suggestions = sorted(suggestions, key=lambda x: calculate_rank(x), reverse=True)
    return sorted_suggestions[:max_suggestions]


def page_rank(graph, damping_factor=0.85, max_iterations=100, tol=1.0e-6):
    n = graph.vertex_count()
    ranks = [1.0 / n] * n
    new_ranks = [0] * n
    
    vertices = list(graph.vertices())
    vertex_indices = {v: i for i, v in enumerate(vertices)}
    
    for _ in range(max_iterations):
        for v in vertices:
            rank_sum = 0
            for e in graph.incident_edges(v, outgoing=False):
                u = e.opposite(v)
                rank_sum += ranks[vertex_indices[u]] / graph.degree(u, incoming=False)
            
            new_ranks[vertex_indices[v]] = (1 - damping_factor) / n + damping_factor * rank_sum
        
        if sum(abs(new_ranks[i] - ranks[i]) for i in range(n)) < tol:
            break
        ranks, new_ranks = new_ranks, ranks
    
    return ranks

def calculate_score(result, page_ranks, term_counts, alpha=0.5):
    page_index = result.page
    if page_index < len(page_ranks):
        rank_score = page_ranks[page_index]
    else:
        rank_score = 0 

    if page_index < len(term_counts):
        term_score = term_counts[page_index]
    else:
        term_score = 0  

    return alpha * rank_score + (1 - alpha) * term_score

def count_word_occurrences(graph, search_term):
    term_counts = []
    for page in graph.vertices():
        if page.element() and isinstance(page.element(), dict):
            text = page.element().get('text', '').lower()
            words = text.split()
            term_count = sum(1 for word in words if word == search_term.lower())
            term_counts.append(term_count)
        else:
            term_counts.append(0)  
    return term_counts

def load_page_text(page_number):
    file_path = f"./txts/page_{page_number}.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    return ""

def search_phrase(phrase, trie_structure, graph=None, context_sentences=2):
    split_words = phrase.split()
    if len(split_words) < 2:
        print("Fraza bi trebala da ima bar 2 reči, ali pretraga će biti izvršena po 1 koju ste uneli.")
        search_multiple_words(split_words, trie_structure, graph, context_sentences)
        return

    initial_results = trie_structure.search(split_words[0])
    if not initial_results:
        print(f"Nema rezultata pretrage za frazu '{phrase}'.")
        return

    term_counts = {term: count_word_occurrences(graph, term) for term in split_words} if graph else {}

    final_results = []
    for result in initial_results:
        page_text = load_page_text(result.page)
        if page_text:
            words = page_text.split()
            for i in range(len(words) - len(split_words) + 1):
                if words[i:i + len(split_words)] == split_words:
                    rank_score = calculate_score(result, page_ranks=page_rank(graph), term_counts=term_counts)
                    final_results.append((rank_score, result.page))
                    break

    if not final_results:
        print(f"Nema rezultata pretrage za frazu '{phrase}'.")
    else:
        final_results.sort(key=lambda x: x[0], reverse=True)
        search_and_display_results(final_results, [phrase], context_sentences=context_sentences)


def parse_logical_expression(query):
    logical_operators = ["AND", "OR", "NOT"]
    stack = []
    result = []

    i = 0
    while i < len(query):
        token = query[i]

        if token == "(":
            stack.append(token)
        elif token == ")":
            while stack and stack[-1] != "(":
                result.append(stack.pop())
            stack.pop()
        elif token in logical_operators:
            while stack and stack[-1] != "(" and logical_operators.index(stack[-1]) >= logical_operators.index(token):
                result.append(stack.pop())
            stack.append(token)
        elif token.startswith('"'):
            phrase = []
            while i < len(query) and not query[i].endswith('"'):
                phrase.append(query[i])
                i += 1
            if i < len(query):
                phrase.append(query[i])
            result.append(' '.join(phrase))
        else:
            result.append(token)

        i += 1

    while stack:
        result.append(stack.pop())

    return result

def search_query(query, trie_structure, graph=None, context_sentences=2, txt_directory=None):
    query_parts = parse_logical_expression(query.split())
    
    if txt_directory and not graph:
        g, pages_text = build_reference_graph(txt_directory)
    elif graph:
        g = graph
    else:
        print("Error: Neki od podataka fale.")
        return

    logical_operators = ["AND", "OR", "NOT"]

    stack = []
    for token in query_parts:
        print(f"Obrađuje se token: {token}")
        if '*' in token:
            prefix = token[:-1]
            suggestions = trie_structure.get_words_from_prefix(prefix)
            print(f"Autocomplete predlozi za prefiks '{prefix}':")
            counter = 1
            printed_suggestions = set()
            for suggestion in suggestions:
                if suggestion.endswith(")") or suggestion.endswith("]") or suggestion.endswith("-") or suggestion.endswith(".") or suggestion.endswith(",") or suggestion.endswith(":") or suggestion.endswith(";") or suggestion.endswith("!") or suggestion.endswith("?") or suggestion.endswith("'") or suggestion.endswith('"') or suggestion.endswith("}") or suggestion.endswith("»") or suggestion.endswith(")"):
                    if suggestion[:-1] in printed_suggestions:
                        continue
                    print(f"{str(counter)}.   {suggestion[:-1]}*")
                    counter += 1
                    continue

                if suggestion.endswith(".)") or suggestion.endswith(".*") or suggestion.endswith(",'") or suggestion.endswith(".*") or suggestion.endswith('!"'):
                    if suggestion[:-2] in printed_suggestions:
                        print("lol ovo je bilo " + suggestion)
                        continue
                    print(f"{str(counter)}.   {suggestion[:-2]}*")
                    counter += 1
                    continue

                print(str(counter) + ".  " + suggestion)
                counter += 1
                printed_suggestions.add(suggestion)
                if counter >= 5:
                    break
            return
        elif isinstance(token, str) and token.startswith('"') and token.endswith('"'):
            phrase = token[1:-1]
            print(f"Pretražuje se po frazi: {phrase}")
            search_phrase(phrase, trie_structure, graph=g, context_sentences=context_sentences)
            return
        elif token in logical_operators:
            if token == "NOT":
                if stack:
                    operand1 = stack.pop()
                    operand2 = stack.pop()
                    if isinstance(operand1, list) and len(operand2) == 1:
                        search_not(operand1, operand2, trie_structure, graph=g, context_sentences=context_sentences)

            elif token == "AND":
                right_query = stack.pop() if stack else []
                left_query = stack.pop() if stack else []
                if isinstance(left_query, list) and isinstance(right_query, list):
                    combined_query = left_query + right_query
                elif isinstance(left_query, list):
                    combined_query = left_query + [right_query]
                elif isinstance(right_query, list):
                    combined_query = [left_query] + right_query
                else:
                    combined_query = [left_query, right_query]
                search_and(combined_query, trie_structure, graph=g, context_sentences=context_sentences)
            elif token == "OR":
                right_query = stack.pop() if stack else []
                left_query = stack.pop() if stack else []
                if isinstance(left_query, list) and isinstance(right_query, list):
                    combined_query = left_query + right_query
                elif isinstance(left_query, list):
                    combined_query = left_query + [right_query]
                elif isinstance(right_query, list):
                    combined_query = [left_query] + right_query
                else:
                    combined_query = [left_query, right_query]
                search_or(combined_query, trie_structure, graph=g, context_sentences=context_sentences)
        else:
            stack.append(token)

    if stack:
        if isinstance(stack[0], list):
            search_multiple_words(stack[0], trie_structure, graph=g, context_sentences=context_sentences)
        else:
            search_multiple_words(stack, trie_structure, graph=g, context_sentences=context_sentences)
