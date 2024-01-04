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

const formatMessage = (results) => {
  return {
    role: 'user',
    content: `
        Answer the user question using the provided context. Do not preface your answer.      
        Question: Make up a new idea for simple sabotage for office workers in 15 words or less.
        Context: ${results.map((r) => r.pageContent).join('\n')}`,
  };
};

const newMessage = async (history, message) => {
  const results = await openAi.chat.completions.create({
    model: 'gpt-3.5-turbo',
    temperature: 1.2,
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
        Be succinct.
        Do not preface your answer.
        Use an active voice.
        Do not use in your answer things like "One idea", "New Idea", "One new idea" or "A new idea".
        Do not mention previous answers or reuse copy from previous answers.
        `,
    },
  ];

  const store = await loadStore();

  const start = () => {
    rl.question('\nPress Enter to Generate A New Rule: ', async (userInput) => {
      const lowerCase = userInput.toLocaleLowerCase();
      if (lowerCase === 'exit' || lowerCase === 'exit;') {
        rl.close();
        return;
      }

      const results = await store.similaritySearch(question, 2);

      const formattedMessage = formatMessage(results);

      const res = await newMessage(history, formattedMessage);

      history.push(
        {
          ...formattedMessage,
          content: formattedMessage.content.slice(
            0,
            formattedMessage.content.indexOf('Context:')
          ),
        },
        res
      );
      console.log(`\n${res.content}`);

      start();
    });
  };

  start();
};

console.log('Chatbot initialized. Type "exit" to end the chat.');
chat();
