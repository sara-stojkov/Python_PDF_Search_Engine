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
