import OpenAI from 'openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { CharacterTextSplitter } from 'langchain/text_splitter';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import readline from 'node:readline';

export const openAi = new OpenAI({
  apiKey: process.env.OPEN_API_KEY,
});

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const question = process.argv[2] || 'hi';

const createStore = (docs) =>
  MemoryVectorStore.fromDocuments(docs, new OpenAIEmbeddings());

const docsFromPdf = () => {
  const loader = new PDFLoader('simple-sab.pdf');

  return loader.load(
    new CharacterTextSplitter({
      separator: '. ',
      chunkSize: 2500,
      chunkOverlap: 200,
    })
  );
};

const loadStore = async () => {
  const p = await docsFromPdf();

  return createStore(p);
};

const formatMessage = (question, results) => {
  return {
    role: 'user',
    content: `
        Answer the user question using the provided context.        
        Question: ${question}
        Context: ${results.map((r) => r.pageContent).join('\n')}`,
  };
};

const newMessage = async (history, message) => {
  const results = await openAi.chat.completions.create({
    model: 'gpt-4',
    temperature: 0,
    messages: [...history, message],
  });

  return results.choices[0].message;
};

const chat = async () => {
  const history = [
    {
      role: 'assistant',
      content: `
        You are a helpful AI assistant.
        Answer the question to the best of your ability.
        Feel free to be creative.
        Do not preface your answer with things like "New Idea" or "A new idea for simple sabotage could be".
        Do not mention previous answers.
        `,
    },
    {
      role: 'user',
      content: `Answer the user's questions using the provided context.
        Question: ${question}`,
    },
  ];

  const store = await loadStore();

  const start = () => {
    rl.question('\nYou: ', async (userInput) => {
      const lowerCase = userInput.toLocaleLowerCase();
      if (lowerCase === 'exit' || lowerCase === 'exit;') {
        rl.close();
        return;
      }

      const results = await store.similaritySearch(question, 2);

      const formattedMessage = formatMessage(userInput, results);

      const res = await newMessage(history, formattedMessage);

      history.push(formattedMessage, res);

      console.log(`\n${res.content}`);

      start();
    });
  };

  start();
};

console.log('Chatbot initialized. Type "exit" to end the chat.');
chat();
