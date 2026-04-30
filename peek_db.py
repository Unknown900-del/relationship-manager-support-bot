import chromadb

client = chromadb.PersistentClient(path="./chromadb")
collection = client.get_collection(name="banking_bot")

# Get all metadatas to see the variety
results = collection.get()
metadatas = results['metadatas']

# Count how many chunks per bank
bank_counts = {}
for meta in metadatas:
    bank = meta.get('bank', 'Unknown')
    bank_counts[bank] = bank_counts.get(bank, 0) + 1

print("--- Bank Distribution ---")
for bank, count in bank_counts.items():
    print(f"{bank}: {count} chunks")