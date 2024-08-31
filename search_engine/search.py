import os
import re
from collections import defaultdict
from difflib import get_close_matches
from structures.trie import NodeInfo
from search_engine.load_parse import load_page_text, build_reference_graph

class NumberOfParametersException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def highlight_term(text, term):
    highlighted_text = re.sub(f'(?i)({re.escape(term)})', f"\033[1;32;40m\\1\033[0m", text)
    return highlighted_text


def get_first_occurrence_context(text, term, context_sentences=2):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    context = []

    for i, sentence in enumerate(sentences):
        if term.lower() in sentence.lower():
            start_index = max(0, i - context_sentences)
            end_index = min(len(sentences), i + context_sentences + 1)
            context = sentences[start_index:end_index]
            break

    if context:
        return ' '.join(context) 
    else:
        lines = text.splitlines()
        for line in lines:
            if term.lower() in line.lower():
                return line.strip()

    return "NEMA"

def search_and_display_results(results, terms, start_rank=1, results_per_page=5, context_sentences=2):
    if not results:
        print("Nema rezultata za pretragu.")
        return

    rank = start_rank
    total_results = len(results)
    page_offset = 0

    results.sort(key=lambda x: x[0], reverse=True)

    first_10_results = []
    for i in range(min(10, total_results)):
        result = results[i]
        if isinstance(result, NodeInfo):
            page_index = result.page
        elif isinstance(result, tuple) and len(result) == 2:
            page_index = result[1]
        else:
            continue
        first_10_results.append(page_index)



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
                return first_10_results

        else:
            return first_10_results

def search_one_word(term, trie_structure, graph, context_sentences=2):
    term=term[0]
    search_results = {}
    # print (f"Pretraga za reč '{term}'")

    results_from_trie = trie_structure.search(term)
    if results_from_trie:
        for result in results_from_trie:
            page_index = result.page
            if page_index in search_results:
                search_results[page_index]['nodes'].append(result)
            else:
                search_results[page_index] = {'nodes': [result]}

    if not search_results:
        print(f"Nema rezultata za '{term}'.")
        did_you_mean(trie_structure, term)
        return

    page_ranks = page_rank(graph, damping_factor=0.85) if graph else [0] * len(graph.vertices())

    term_counts = count_word_occurrences(graph, term) if graph else {}

    combined_results = []

    for page_index, data in search_results.items():
        nodes = data['nodes']
        node = nodes[0]
        score = calculate_score(graph, node, page_ranks, term_counts)

        combined_results.append((score, page_index))

    combined_results.sort(key=lambda x: x[0], reverse=True)
    if len(combined_results) < 1:
        did_you_mean(trie_structure, term)
    else:
        return search_and_display_results(combined_results, [term], context_sentences=context_sentences)

def search_multiple_words(query, trie_structure, graph, context_sentences=2):
    search_results = {}
    for term in query:
        results_from_trie = trie_structure.search(term)
        if results_from_trie:
            for result in results_from_trie:
                page_index = result.page
                if page_index in search_results:
                    search_results[page_index]['terms'].add(term)
                    search_results[page_index]['nodes'].append(result)
                else:
                    search_results[page_index] = {'terms': {term}, 'nodes': [result]}
    
    if not search_results:
        print(f"Nema rezultata za '{' '.join(query)}'.")
        return

    page_ranks = page_rank(graph, damping_factor=0.85) if graph else [0] * len(graph.vertices())
    # print(page_ranks)

    term_counts = {term: count_word_occurrences(graph, term) for term in query} if graph else {}
    combined_results = []

    for page_index, data in search_results.items():
        terms = data['terms']
        nodes = data['nodes']
        
        total_score = sum(calculate_score(graph, node, page_ranks, term_counts) for node in nodes)
        
        if set(query).issubset(terms):
            total_score *= 2

        combined_results.append((total_score, page_index))

    combined_results.sort(key=lambda x: x[0], reverse=True)
    
    return search_and_display_results(combined_results, query, context_sentences=context_sentences)



def search_and(query, trie_structure, graph=None, context_sentences=2):
    term_results = [trie_structure.search(term) for term in query]
    page_count = {}
    for result_list in term_results:
        unique_pages = set()
        for result in result_list:
            page_number = result.page
            if page_number not in unique_pages:
                if page_number not in page_count:
                    page_count[page_number] = 1
                else:
                    page_count[page_number] += 1
                unique_pages.add(page_number)

    common_pages = set()
    for page, count in page_count.items():
        if count == len(query):
            # print(f"Page: {page}, Count: {count}")
            common_pages.add(page)

    print(f"Broj zajedničkih stranica: {len(common_pages)}")

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
                        combined_score += calculate_score(graph, result, page_ranks, term_counts)
            combined_results.append((combined_score, page))
        combined_results.sort(key=lambda x: x[0], reverse=True)
        return search_and_display_results(combined_results, query, context_sentences=context_sentences)


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
            score = sum(calculate_score(graph, node, page_ranks, term_counts.get(term, [])) for node, term in zip(nodes, terms))
            
            combined_results.append((score, page_index))

        combined_results.sort(key=lambda x: x[0], reverse=True)
        return search_and_display_results(combined_results, query, context_sentences=context_sentences)

def search_not(query1, query2, trie_structure, graph=None, context_sentences=2):
    term_results1 = [trie_structure.search(query1)]
    term_results2 = [trie_structure.search(query2)]

    page_count = {}
    for result_list in term_results1:
        for result in result_list:
            page_number = result.page
            if page_number not in page_count:
                page_count[page_number] = 1
            
    for result_list in term_results2:
        for result in result_list:
            page_number = result.page
            if page_number not in page_count:
                page_count[page_number] = 5
            else:
                page_count[page_number] += 1
    

    common_pages = set()
    for page, count in page_count.items():
        if count == 2:
            # print(f"Page: {page}, Count: {count}")
            common_pages.add(page)

    # print(f"Number of common Pages: {len(common_pages)}")

    valid_pages = {page for page, count in page_count.items() if count == 1}
    print(f"Broj validnih stranica: {len(valid_pages)}")
    # exclude_pages = {page for page, count in page_count.items() if count == 2}
    # valid_pages = query1_pages - exclude_pages

    if not valid_pages:
        print(f"Nema rezultata za '{query1}' koji ne sadrže '{query2}'.")
        return
    page_ranks = page_rank(graph) if graph else [0] * len(graph.vertices())
    damdaram=[query1]
    term_counts = {term: count_word_occurrences(graph, term) for term in damdaram} if graph else {}
    combined_results = []
    for page in valid_pages:
        combined_score = 0
        nodes = []
        for term in damdaram:
            results = trie_structure.search(term)
            for result in results:
                if result.page == page:
                    nodes.append(result)
                    combined_score += calculate_score(graph, result, page_ranks, term_counts)
        combined_results.append((combined_score, page))

    combined_results.sort(key=lambda x: x[0], reverse=True)
    return search_and_display_results(combined_results, damdaram, context_sentences=context_sentences)


def suggest_alternative_queries(query, trie_structure, max_suggestions=5):
    def get_all_words_from_trie(trie):
        def dfs(node, prefix):
            words = []
            if node.is_word_end:
                words.append(prefix)
            for char, child in node.children.items():
                words.extend(dfs(child, prefix + char))
            return words
        
        return dfs(trie.root, "")
    
    def calculate_rank(query_term):
        search_results = trie_structure.search(query_term)
        if not search_results:
            return 0
        return sum(len(result.positions) for result in search_results)  
    
    all_words = get_all_words_from_trie(trie_structure)
    
    suggestions = get_close_matches(query, all_words, n=max_suggestions)
    sorted_suggestions = sorted(suggestions, key=lambda x: calculate_rank(x), reverse=True)

    return sorted_suggestions


def did_you_mean(trie, query, max_suggestions=3):
    """Ukoliko rezultata pretrage nema (nema rangiranih stranica) ili se zadati 
    upit pojavljuje na malom broju stranica, ponuditi korisniku da zadati upit zameni 
    sličnim, popularnijim upitom."""
    suggestions = suggest_alternative_queries(query, trie, max_suggestions)

    if suggestions:
        print("\nDa li ste mislili:")
        for suggestion in suggestions:
            print(f" - {suggestion}")
    else:
        print("Nema predloga za alternativne upite.")


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


def calculate_score(graph, result, page_ranks, term_counts):    
    page_index = result.page    
    rank_score = page_ranks[page_index] if page_index < len(page_ranks) else 0

    vertex = graph.find_vertex_by_index(page_index)

    term_score = result.occurrences if vertex is not None else 0
    
    score = term_score + rank_score 
    # print(f"Page Index: {page_index}" , score )
    return score

def count_word_occurrences(graph, term):
    # print(f"Broj pojavljivanja reči '{term}' na stranicama:")
    term = term.lower()
    print(f"Term: {term}")
    term_counts = defaultdict(int)

    word_pattern = re.compile(r'\b\w+\b')

    for page_number in graph.vertices():
        text = load_page_text(page_number)
        if text:
            text = text.lower()
            words = word_pattern.findall(text)
            occurrences = words.count(term)
            if occurrences > 0:
                # print(f"Strana: {page_number}, Broj pojavljivanja: {occurrences}")
                term_counts[page_number] += occurrences

    return term_counts


def load_page_text(page_number):
    file_path = f"./txts/page_{page_number}.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    return ""

def search_phrase(phrase, trie_structure, graph=None, context_sentences=2):
    split_words = phrase.lower().split()
    if len(split_words) < 2:
        search_one_word(split_words, trie_structure, graph, context_sentences)
        return

    initial_results = trie_structure.search(split_words[0])
    if not initial_results:
        print(f"Nema rezultata pretrage za frazu '{phrase}'.")
        return

    term_counts = {term: count_word_occurrences(graph, term) for term in split_words} if graph else {}
    final_results = []
    passed_pages = set()
    for result in initial_results:
        page_text = load_page_text(result.page)
        if result.page in passed_pages:
            continue
        if page_text:
            words = re.findall(r'\b\w+\b', page_text.lower())
            for i in range(len(words) - len(split_words) + 1):
                if words[i:i + len(split_words)] == split_words:
                    rank_score = calculate_score(graph, result, page_ranks=page_rank(graph), term_counts=term_counts)
                    # print(f"{rank_score} je skor za stranu {result.page}")
                    final_results.append((rank_score, result.page))
                    passed_pages.add(result.page)
                    break


    if not final_results:
        print(f"Nema rezultata pretrage za frazu '{phrase}'.")
    else:
        sorted_results = sorted(final_results, key=lambda x: x[0], reverse=True)

        print(f"Rezultati pretrage za frazu '{phrase}':")
        
        return search_and_display_results(sorted_results, [phrase], context_sentences=context_sentences)

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
                if suggestion in printed_suggestions:
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
            return search_phrase(phrase, trie_structure, graph=g, context_sentences=context_sentences)
        elif token in logical_operators:
            if token == "NOT":
                if stack:
                    operand2 = stack.pop()
                    operand1 = stack.pop()
                
                    return search_not(operand1, operand2, trie_structure, graph=g, context_sentences=context_sentences)
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
                return search_and(combined_query, trie_structure, graph=g, context_sentences=context_sentences)
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
                return search_or(combined_query, trie_structure, graph=g, context_sentences=context_sentences)
        else:
            stack.append(token)

    if stack:
        if isinstance(stack[0], list):
            if len(stack) == 1:
                return search_one_word(stack[0], trie_structure, graph=g, context_sentences=context_sentences)
            else:
                return search_multiple_words(stack[0], trie_structure, graph=g, context_sentences=context_sentences)
        else:
            if len(stack) == 1:
                return search_one_word(stack, trie_structure, graph=g, context_sentences=context_sentences)
            else:
                return search_multiple_words(stack, trie_structure, graph=g, context_sentences=context_sentences)
    
    return []