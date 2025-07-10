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
<section id="upload" className="py-20 px-6 bg-gray-50">
      <div className="max-w-xl mx-auto bg-white p-8 rounded-2xl shadow-lg text-center">
        <h2 className="text-3xl font-bold mb-4 text-gray-800">
          Upload and Convert
        </h2>
        <p className="text-gray-500 mb-6">
          Upload your PDF or image file and convert it into editable Word format.
        </p>

        <label className="cursor-pointer bg-indigo-50 border-2 border-dashed border-indigo-200 p-6 rounded-xl block hover:bg-indigo-100 transition mb-6">
          <input
            type="file"
            accept=".pdf,.jpg,.jpeg,.png"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="hidden"
          />
          {file ? (
            <span className="text-indigo-600 font-medium">{file.name}</span>
          ) : (
            <span className="text-indigo-400">Click to upload or drag a file here</span>
          )}
        </label>

        <button
          onClick={handleUpload}
          disabled={!file || isLoading}
          className="bg-indigo-600 text-white px-6 py-3 rounded-full font-semibold hover:bg-indigo-700 disabled:opacity-50 transition w-full"
        >
          {isLoading ? "Converting..." : "Convert to Word"}
        </button>
      </div>
    </section>
  );
}
