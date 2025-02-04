from rag_app import RAGApplication
import argparse


def main():
    parser = argparse.ArgumentParser(description="RAG Application with PDF support")
    parser.add_argument("--pdf", type=str, help="Path to PDF file to load")
    args = parser.parse_args()

    rag = RAGApplication()

    # Try to load existing vector store
    if not args.pdf and rag.load_existing_vector_store():
        print("Loaded existing vector store")
    elif args.pdf:
        print(f"Processing PDF: {args.pdf}")
        num_chunks = rag.load_pdf(args.pdf)
        print(f"Processed {num_chunks} text chunks")
    else:
        print("Please provide a PDF file or ensure there's an existing vector store")
        return

    print("\nRAG System Ready! Enter your questions (type 'exit' to quit)")

    while True:
        question = input("\nQuestion: ")
        if question.lower() == "exit":
            break

        try:
            answer = rag.query(question)
            print("\nAnswer:", answer)
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
