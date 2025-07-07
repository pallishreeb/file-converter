export default function Hero() {
  return (
    <section className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-20 px-4 text-center">
      <h1 className="text-4xl md:text-6xl font-bold mb-4">
        AI-Powered File Conversion & Chat with Documents
      </h1>
      <p className="text-xl max-w-2xl mx-auto">
        Convert images, PDFs, Word docs, and even chat with them using AI. All in one simple tool.
      </p>
      <a
        href="#upload"
        className="mt-8 inline-block bg-white text-indigo-600 px-6 py-3 rounded-full font-semibold hover:bg-gray-100 transition"
      >
        Try it Now
      </a>
    </section>
  );
}
