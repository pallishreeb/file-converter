import Image from "next/image";
import Hero from "@/components/Hero";
import Features from "@/components/Features";
import UploadCTA from "@/components/UploadCTA";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <main className="min-h-screen bg-white text-gray-900">
      <Hero />
      <Features />
      <UploadCTA />
      <Footer />
    </main>
  );
}
