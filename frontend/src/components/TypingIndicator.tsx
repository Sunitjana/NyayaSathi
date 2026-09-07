"use client";

export default function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 px-4 py-3">
      <span className="w-2 h-2 rounded-full bg-indigo-400 typing-dot" />
      <span className="w-2 h-2 rounded-full bg-indigo-400 typing-dot" />
      <span className="w-2 h-2 rounded-full bg-indigo-400 typing-dot" />
      <span className="ml-2 text-sm text-gray-400">Analyzing legal documents...</span>
    </div>
  );
}
