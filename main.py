import search_engine.load_parse as lp
import search_engine.search as s
import search_engine.additions as add

def main(file_path):
    print("----------------------------------\n  Pretrazivač teksta u PDF fajlu\n---------------------------------\n")
    print(f"Trenutno se učitava PDF fajl '{file_path}'...")
    trie_structure, graph, vertices = lp.load_and_parse(file_path)
    if trie_structure is None:
        print("Neuspešno učitavanje PDF-a. Izlazak iz aplikacije...")
        return
    
    while True:
        print("\n----------------------------------")
        print("Unesite reč* za autocomplete, ili koristeći AND, OR, NOT precizirajte logičko pretraživanje.")
        query = input("Unesite reči koje želite pretražiti ili unesite x za izlazak: ")
        if query.lower() == 'x':
            break
        elif query == "":
            print("Molimo unesite reči za pretragu.")
            continue
        elif query in {"AND", "OR", "NOT"}:
            print("Reči AND, NOT i OR su rezervisane za logičko pretraživanje.")
            continue
        print(f"Pretraživanje po upitu '{query}'...")
        s.search_query(query, trie_structure, graph, 2, txt_directory="txts")
        # save_to_pdf = input("Da li želite da sačuvate prvih 10 rezultata pretrage u zaseban PDF? (unesite 'DA' ili 'NE')  --->  ")
        # if save_to_pdf == "DA":
        #     add.save_results_to_pdf(text_output)
        highlight_the_word = input("Da li želite da highlightujete ovu reč u celom dokumentu? (unesite 'DA' ili 'NE')  --->  ")
        if highlight_the_word == "DA":
            add.highlight_word_in_pdf

if __name__ == "__main__":
    main("./pdf/Data Structures and Algorithms in Python.pdf")
    print("\n----------------------------------")
    print("Hvala na korišćenju pretraživača teksta u PDF fajlu!")
