const features = [
  { title: "Convert Any File", desc: "Image, Word, PDF – convert effortlessly." },
  { title: "Chat with Your File", desc: "Ask questions. Get answers instantly." },
  { title: "Fast & Secure", desc: "No data stored. Processed in real-time." },
];

export default function Features() {
  return (
    <section className="py-16 px-6 bg-gray-100 text-center">
      <h2 className="text-3xl font-bold mb-10">What You Can Do</h2>
      <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
        {features.map((f, i) => (
          <div key={i} className="bg-white rounded-2xl p-6 shadow hover:shadow-md transition">
            <h3 className="text-xl font-semibold mb-2">{f.title}</h3>
            <p className="text-gray-600">{f.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
