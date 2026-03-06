import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { CohereEmbeddings } from "@langchain/cohere";
import { CustomPDFLoader } from "@/utils/customPDFLoader";
import { DirectoryLoader } from "@langchain/community/document_loaders/fs/directory";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { COHERE_API_KEY } from "@/config/cohere";
import { COLLECTION_NAME } from "@/config/chroma";

/* Name of directory to retrieve your files from */
const filePath = "docs";

export const run = async () => {
  try {
    /* Load raw docs from all files in the directory */
    const directoryLoader = new DirectoryLoader(filePath, {
      ".pdf": (path) => new CustomPDFLoader(path),
    });

    const rawDocs = await directoryLoader.load();

    /* Split text into chunks */
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const docs = await textSplitter.splitDocuments(rawDocs);
    console.log("split docs", docs);

    console.log("creating vector store...");

    /* Create and store the embeddings in the vector store */
    const embeddings = new CohereEmbeddings({
      model: "embed-english-v3.0",
      apiKey: COHERE_API_KEY,
      batchSize: 48,
    });

    // Optional: clear existing collection first
    const chroma = new Chroma(embeddings, {
      collectionName: COLLECTION_NAME,
    });

    if (chroma.index?.reset) {
      await chroma.index.reset();
    }

    // Ingest documents in batches of 100
    for (let i = 0; i < docs.length; i += 100) {
      const batch = docs.slice(i, i + 100);

      await Chroma.fromDocuments(batch, embeddings, {
        collectionName: COLLECTION_NAME,
      });
    }

    console.log("Ingestion of documents complete");
  } catch (error) {
    console.error("error", error);
    throw new Error("Failed to ingest your data");
  }
};

(async () => {
  await run();
})();
