"use client";
import { useState } from "react";

export default function UploadCTA() {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://localhost:5000/convert", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      alert("Conversion failed.");
      setIsLoading(false);
      return;
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = file.name.replace(/\.[^/.]+$/, "") + ".docx";
    document.body.appendChild(link);
    link.click();
    link.remove();
    setIsLoading(false);
  };

  return (
    <section id="upload" className="py-20 px-6 text-center bg-white">
      <h2 className="text-3xl font-bold mb-6">Upload and Convert</h2>
      <input
        type="file"
        accept=".pdf,.jpg,.jpeg,.png"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
        className="block mx-auto mb-4"
      />
      <button
        onClick={handleUpload}
        disabled={!file || isLoading}
        className="bg-indigo-600 text-white px-6 py-3 rounded-full font-semibold hover:bg-indigo-700 transition"
      >
        {isLoading ? "Converting..." : "Convert to Word"}
      </button>
    </section>
  );
}
