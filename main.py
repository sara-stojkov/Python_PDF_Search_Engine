import load_parse as lp
import search as s
from structures import trie

def main(file_path):
    print("------------------------------\n  Pretrazivač teksta u PDF fajlu\n------------------------------\n")
    print(f"Trenutno se učitava PDF fajl '{file_path}'...")
    trie_structure = lp.load_and_parse(file_path)
    if trie_structure is None:
        print("Failed to load the PDF. Exiting...")
        return
    
    while True:
        query = str(input("Unesite reči koje želite pretražiti ili unesite x za izlazak:  "))
        if query == "x":
            break

        elif query == "":
            print("Molimo unesite reči za pretragu.")
            continue
        elif query == "AND" or query == "OR" or query == "NOT":
            print("Reči AND, NOT i OR su rezervisane za logičko pretraživanje.")
            continue
        print(f"Pretraživanje po upitu '{query}'...")
        s.search_query(query, trie_structure)
    print("Hvala na korišćenju pretraživača teksta u PDF fajlu!")

if __name__ == "__main__":
    main("./pdf/Data Structures and Algorithms in Python.pdf")
