import search_engine.load_parse as lp
import search_engine.search as s
import fitz  


def save_results_to_file(search_results, query, file_path):
    """Funkcija koja čuva prvih 10 rezultata pretrage u obliku PDF-a."""
    original_doc = fitz.open(file_path)
    output_doc = fitz.open()

    for page_number in search_results:
        page = original_doc.load_page(page_number - 1)
        output_page = output_doc.new_page(width=page.rect.width, height=page.rect.height)
        output_page.show_pdf_page(output_page.rect, original_doc, page_number - 1)
        if len(query) == 1:
            highlight_query_on_page(output_page, [query])
        else:
            highlight_query_on_page(output_page, query.split())

    output_filename = f"{query.replace(' ', '_')}_search_results.pdf"
    if "AND" in query or "OR" in query or "NOT" in query:
        output_filename = f"{query.replace(' ', '_')}_search_results.pdf"
    try:
        output_doc.save(output_filename)
    except Exception as e:
        print(f"Error saving the file: {e}")
        output_filename = f"{query[1:-1].replace(' ', '_')}__search_results.pdf"
    finally:
        output_doc.close()
        original_doc.close()
    print(f"Search results saved in file '{output_filename}'.")

def highlight_query_on_page(page, queries):
    for query in queries:
        if query=="AND" or query=="OR" or query=="NOT":
            continue
        text_instances = page.search_for(query)
        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)
            highlight.update()

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
        elif query in {"AND", "OR", "NOT", "()", "(", ")"}:
            print("Reči AND, NOT i OR su rezervisane za logičko pretraživanje.")
            continue
        print(f"Pretraživanje po upitu '{query}'...")
        search_results = s.search_query(query, trie_structure, graph, 2, txt_directory="txts")
                
        if search_results is None:
            print("Pošto nije bilo rezultata, nije moguće čuvanje rezultata u PDF.")
            continue

        save_results_to_pdf = input("Da li želite da prvih 10 rezultata pretrage u zasebnom dokumentu? (unesite 'DA' ili 'NE')  --->  ")
        if save_results_to_pdf.upper() == "DA":
            save_results_to_file(search_results, query, file_path) # query će se koristiti za naziv fajla

if __name__ == "__main__":
    main("./pdf/Data Structures and Algorithms in Python.pdf")
    print("\n----------------------------------")
    print("Hvala na korišćenju pretraživača teksta u PDF fajlu!")
    print("************************************")
