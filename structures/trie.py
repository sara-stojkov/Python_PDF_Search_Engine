class NodeInfo:
    def __init__(self, page, positions, occurrences):
        self.page = page
        self.positions = positions
        self.occurrences = occurrences

    def __str__(self):
        return f"Page: {self.page}, Positions: {self.positions}, Occurrences: {self.occurrences}"

class TrieNode:
    def __init__(self, char):
        self.char = char
        self.children = {}
        self.is_word_end = False
        self.info = []

class Trie:
    def __init__(self):
        self.root = TrieNode(None)

    def insert(self, word, info):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode(char)
            node = node.children[char]
        node.is_word_end = True
        node.info.append(info)

    def search(self, prefix):
        node = self.root
        results = []

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        self._collect_info(node, results)
        return results

    def _collect_info(self, node, results):
        if node.is_word_end:
            results.extend(node.info)
        for child in node.children.values():
            self._collect_info(child, results)

    def __str__(self):
        return self._print(self.root)
    
    def find_words_with_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return self._get_words(node, prefix)

    def get_words_from_prefix(self, prefix):
        def dfs(node, prefix):
            words = []
            if node.is_word_end:
                words.append(prefix)
            for char, child in node.children.items():
                words.extend(dfs(child, prefix + char))
            return words

        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
        return dfs(node, prefix)
    
    def search_phrase(self, phrase):
        words = phrase.split()
        results = self.search(words[0])
        if not results:
            return []

        filtered_results = []
        for result in results:
            if result.page == words[0][0]: 
                current_node = result
                for word in words[1:]:
                    next_node = current_node.children.get(word)
                    if not next_node:
                        break
                    current_node = next_node
                if current_node.is_word_end:
                    filtered_results.append(current_node)
        return filtered_results

