"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import {
  Search,
  Scale,
  Send,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  Sparkles,
  BarChart3,
  MessageSquare,
  Gavel,
  Menu,
  X,
  Sun,
  Moon,
  Copy,
  Check,
  Download,
  RefreshCw,
  Wifi,
  WifiOff,
  Keyboard,
} from "lucide-react";
import {
  streamQuery,
  fetchStats,
  checkHealth,
  type StreamChunk,
  type StatsResult,
  type HistoryTurn,
} from "@/lib/api";
import ConfidenceBadge from "@/components/ConfidenceBadge";
import SourceChips from "@/components/SourceChips";
import TypingIndicator from "@/components/TypingIndicator";

const Scene3D = dynamic(() => import("@/components/Scene3D"), { ssr: false });

type Mode = "search" | "assistant";
type Theme = "dark" | "light";

interface Message {
  role: "user" | "ai";
  content: string;
  sources?: string[];
  confidence?: string;
  category?: string;
  chunks?: { text: string; source: string; score: number }[];
  relatedQuestions?: string[];
  error?: boolean;
}

const CATEGORIES = [
  { id: "auto", label: "Auto-detect" },
  { id: "criminal", label: "Criminal" },
  { id: "civil", label: "Civil" },
  { id: "property", label: "Property" },
  { id: "consumer", label: "Consumer" },
  { id: "labor", label: "Labor" },
  { id: "family", label: "Family" },
];

export default function Home() {
  const [mode, setMode] = useState<Mode>("search");
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("auto");
  const [searchMessages, setSearchMessages] = useState<Message[]>([]);
  const [assistantMessages, setAssistantMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [stats, setStats] = useState<StatsResult | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [expandedChunk, setExpandedChunk] = useState<number | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [theme, setTheme] = useState<Theme>("dark");
  const [connected, setConnected] = useState<boolean | null>(null);
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const messages = mode === "search" ? searchMessages : assistantMessages;
  const setMessages = mode === "search" ? setSearchMessages : setAssistantMessages;

  useEffect(() => {
    const saved = localStorage.getItem("legalai-theme") as Theme | null;
    if (saved) setTheme(saved);
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("legalai-theme", theme);
  }, [theme]);

  useEffect(() => {
    setStatsLoading(true);
    fetchStats().then((s) => { setStats(s); setConnected(true); }).catch(() => { setConnected(false); }).finally(() => setStatsLoading(false));
    const interval = setInterval(() => {
      checkHealth().then(setConnected);
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        inputRef.current?.focus();
      }
      if (e.key === "Escape") {
        setQuery("");
        inputRef.current?.blur();
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "1") {
        e.preventDefault();
        setMode("search");
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "2") {
        e.preventDefault();
        setMode("assistant");
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const buildHistory = useCallback((): HistoryTurn[] => {
    return messages
      .filter((m) => m.content && !m.error)
      .slice(-10)
      .map((m) => ({ role: m.role === "user" ? "user" : "assistant", content: m.content }));
  }, [messages]);

  const handleSubmit = useCallback(async (overrideQuery?: string) => {
    const q = (overrideQuery || query).trim();
    if (!q || isStreaming) return;

    if (!overrideQuery) setQuery("");
    const prevMessages = mode === "search" ? searchMessages : assistantMessages;
    const newMessages = [...prevMessages, { role: "user" as const, content: q }];
    setMessages(newMessages);
    setIsStreaming(true);

    let currentAnswer = "";
    let currentSources: string[] = [];
    let currentChunks: { text: string; source: string; score: number }[] = [];
    let currentConfidence = "";
    let currentCategory = "";
    let currentRelated: string[] = [];

    const aiIndex = newMessages.length;
    setMessages([...newMessages, { role: "ai", content: "", sources: [], confidence: "" }]);

    const history = buildHistory();

    try {
      await streamQuery(
        q,
        mode,
        category === "auto" ? undefined : category,
        history,
        (chunk: StreamChunk) => {
          switch (chunk.type) {
            case "category":
              currentCategory = chunk.category;
              break;
            case "sources":
              currentSources = chunk.sources;
              currentChunks = chunk.chunks;
              break;
            case "chunk":
              currentAnswer += chunk.content;
              setMessages((prev) => {
                const updated = [...prev];
                updated[aiIndex] = {
                  role: "ai",
                  content: currentAnswer,
                  sources: currentSources,
                  chunks: currentChunks,
                  category: currentCategory,
                };
                return updated;
              });
              break;
            case "done":
              currentConfidence = chunk.confidence || "medium";
              currentRelated = chunk.related_questions || [];
              setMessages((prev) => {
                const updated = [...prev];
                updated[aiIndex] = {
                  role: "ai",
                  content: currentAnswer,
                  sources: currentSources,
                  confidence: currentConfidence,
                  chunks: currentChunks,
                  category: currentCategory,
                  relatedQuestions: currentRelated,
                };
                return updated;
              });
              break;
          }
        }
      );
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      setMessages((prev) => {
        const updated = [...prev];
        updated[aiIndex] = {
          role: "ai",
          content: `Error: ${errorMsg}`,
          confidence: "low",
          error: true,
        };
        return updated;
      });
    }

    setIsStreaming(false);
  }, [query, mode, category, isStreaming, searchMessages, assistantMessages, setMessages, buildHistory]);

  const handleRetry = useCallback((msgIndex: number) => {
    const userMsg = messages[msgIndex - 1];
    if (!userMsg || userMsg.role !== "user") return;
    setMessages((prev) => prev.slice(0, msgIndex - 1));
    setTimeout(() => handleSubmit(userMsg.content), 100);
  }, [messages, setMessages, handleSubmit]);

  const handleCopy = useCallback((text: string, idx: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 2000);
  }, []);

  const handleExportPDF = useCallback((question: string, answer: string) => {
    const w = window.open("", "_blank");
    if (!w) return;
    w.document.write(`
      <html><head><title>Legal AI - Answer</title>
      <style>body{font-family:system-ui;max-width:800px;margin:40px auto;padding:0 20px;color:#1a1a2e}
      h1{font-size:18px;color:#6366f1}h2{font-size:14px;color:#666;margin-bottom:20px}
      .answer{line-height:1.8;white-space:pre-wrap}.footer{margin-top:40px;padding-top:20px;border-top:1px solid #eee;font-size:12px;color:#999}</style>
      </head><body><h1>NyayaSathi - Your Legal Rights, Made Simple</h1>
      <h2>Question: ${question.replace(/</g, "&lt;")}</h2>
      <div class="answer">${answer.replace(/</g, "&lt;").replace(/\n/g, "<br>")}</div>
      <div class="footer">Generated by NyayaSathi. This is AI-generated legal information, not professional legal advice.</div>
      </body></html>
    `);
    w.document.close();
    w.print();
  }, []);

  const sidebarContent = (
    <>
      <div className="p-6 border-b border-indigo-500/10">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-500/20 flex items-center justify-center">
              <Scale size={22} className="text-indigo-400" />
            </div>
            <div>
              <h1 className="font-bold text-lg glow-text">NyayaSathi</h1>
              <p className="text-xs text-gray-500">Your Legal Rights, Made Simple</p>
            </div>
          </div>
          <button onClick={() => setSidebarOpen(false)} className="md:hidden text-gray-400">
            <X size={20} />
          </button>
        </div>
      </div>

      <div className="p-4 space-y-2">
        <p className="text-xs uppercase tracking-wider text-gray-500 mb-3 font-semibold">Mode</p>
        <button
          onClick={() => { setMode("search"); setSidebarOpen(false); }}
          className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
            mode === "search"
              ? "bg-indigo-500/15 text-indigo-300 border border-indigo-500/30"
              : "text-gray-400 hover:bg-white/5"
          }`}
        >
          <Search size={18} />
          Legal Search
          <span className="ml-auto text-[10px] text-gray-600 hidden md:inline">⌘1</span>
        </button>
        <button
          onClick={() => { setMode("assistant"); setSidebarOpen(false); }}
          className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
            mode === "assistant"
              ? "bg-indigo-500/15 text-indigo-300 border border-indigo-500/30"
              : "text-gray-400 hover:bg-white/5"
          }`}
        >
          <Gavel size={18} />
          Legal Assistant
          <span className="ml-auto text-[10px] text-gray-600 hidden md:inline">⌘2</span>
        </button>
      </div>

      <AnimatePresence>
        {mode === "assistant" && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="px-4 overflow-hidden"
          >
            <p className="text-xs uppercase tracking-wider text-gray-500 mb-2 font-semibold">Category</p>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full bg-white/5 border border-indigo-500/20 rounded-xl px-3 py-2.5 text-sm text-gray-300 focus:outline-none focus:border-indigo-500/50"
            >
              {CATEGORIES.map((c) => (
                <option key={c.id} value={c.id} className="bg-[#0d0d1a]">
                  {c.label}
                </option>
              ))}
            </select>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="px-4 py-3">
        <button
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          className="w-full flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm text-gray-400 hover:bg-white/5 transition-all"
        >
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          {theme === "dark" ? "Light Mode" : "Dark Mode"}
        </button>
      </div>

      <div className="mt-auto p-4 border-t border-indigo-500/10">
        <p className="text-xs uppercase tracking-wider text-gray-500 mb-3 font-semibold flex items-center gap-1.5">
          <BarChart3 size={12} /> Database
        </p>
        {statsLoading ? (
          <div className="space-y-2">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-4 bg-white/5 rounded animate-pulse" />
            ))}
          </div>
        ) : stats ? (
          <div className="space-y-2 text-xs text-gray-400">
            <div className="flex justify-between">
              <span>Total Chunks</span>
              <span className="text-indigo-300 font-mono">{stats.total_chunks.toLocaleString()}</span>
            </div>
            {Object.entries(stats.collections).map(([name, count]) => (
              <div key={name} className="flex justify-between">
                <span className="truncate mr-2">{name.replace("legal_", "")}</span>
                <span className="font-mono text-gray-500">{count.toLocaleString()}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-gray-600">Could not connect to backend</p>
        )}
      </div>

      <div className="p-4 border-t border-indigo-500/10 flex items-center justify-between">
        <div className="flex items-center gap-2">
          {connected === true && <Wifi size={12} className="text-emerald-400" />}
          {connected === false && <WifiOff size={12} className="text-red-400" />}
          {connected === null && <Wifi size={12} className="text-gray-600 animate-pulse" />}
          <span className="text-[10px] text-gray-600">
            {connected === true ? "Connected" : connected === false ? "Disconnected" : "Checking..."}
          </span>
        </div>
        <span className="text-[10px] text-gray-600 flex items-center gap-1">
          <Keyboard size={10} /> ⌘K
        </span>
      </div>
    </>
  );

  return (
    <>
      <Scene3D />

      <div className="flex min-h-screen">
        {/* Desktop Sidebar */}
        <aside className="hidden md:flex w-72 fixed left-0 top-0 h-screen glass border-r border-indigo-500/10 flex-col z-20">
          {sidebarContent}
        </aside>

        {/* Mobile Sidebar Overlay */}
        <AnimatePresence>
          {sidebarOpen && (
            <>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-black/60 z-30 md:hidden"
                onClick={() => setSidebarOpen(false)}
              />
              <motion.aside
                initial={{ x: -288 }}
                animate={{ x: 0 }}
                exit={{ x: -288 }}
                transition={{ type: "spring", damping: 25, stiffness: 300 }}
                className="fixed left-0 top-0 h-screen w-72 glass border-r border-indigo-500/10 flex flex-col z-40 md:hidden"
              >
                {sidebarContent}
              </motion.aside>
            </>
          )}
        </AnimatePresence>

        {/* Main Content */}
        <main className="flex-1 md:ml-72 flex flex-col h-screen">
          <header className="glass border-b border-indigo-500/10 px-4 md:px-8 py-4 md:py-5">
            <div className="flex items-center gap-3">
              <button onClick={() => setSidebarOpen(true)} className="md:hidden text-gray-400">
                <Menu size={22} />
              </button>
              {mode === "search" ? (
                <>
                  <Search size={24} className="text-indigo-400 hidden md:block" />
                  <div>
                    <h2 className="text-lg md:text-xl font-bold">Legal Document Search</h2>
                    <p className="text-xs md:text-sm text-gray-500 hidden sm:block">
                      Ask any question about Indian law — IPC, BNS, Constitution & more
                    </p>
                  </div>
                </>
              ) : (
                <>
                  <Gavel size={24} className="text-indigo-400 hidden md:block" />
                  <div>
                    <h2 className="text-lg md:text-xl font-bold">Personal Legal Assistant</h2>
                    <p className="text-xs md:text-sm text-gray-500 hidden sm:block">
                      Describe your legal problem and get structured guidance
                    </p>
                  </div>
                </>
              )}
            </div>
          </header>

          <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6 space-y-6">
            {messages.length === 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex flex-col items-center justify-center h-full text-center"
              >
                <div className="w-20 h-20 rounded-2xl bg-indigo-500/10 flex items-center justify-center mb-6">
                  <Sparkles size={36} className="text-indigo-400 animate-pulse-slow" />
                </div>
                <h3 className="text-xl md:text-2xl font-bold mb-2 glow-text">
                  {mode === "search" ? "What legal question do you have?" : "Describe your legal problem"}
                </h3>
                <p className="text-gray-500 max-w-md mb-8 text-sm">
                  {mode === "search"
                    ? "Search across IPC, BNS 2023, Constitution, CrPC, and 24,000+ legal Q&A pairs"
                    : "Get your rights, applicable laws, and step-by-step recommended actions"}
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-lg">
                  {(mode === "search"
                    ? [
                        "What is the punishment for theft under IPC?",
                        "Difference between IPC and BNS 2023",
                        "What are fundamental rights?",
                        "Bail provisions in criminal cases",
                      ]
                    : [
                        "My landlord won't return security deposit",
                        "I was wrongfully terminated from my job",
                        "Neighbor encroaching on my property",
                        "Defective product caused injury",
                      ]
                  ).map((example) => (
                    <button
                      key={example}
                      onClick={() => handleSubmit(example)}
                      className="text-left text-sm px-4 py-3 rounded-xl glass glass-hover text-gray-400 hover:text-indigo-300 transition-all"
                    >
                      <MessageSquare size={14} className="inline mr-2 opacity-50" />
                      {example}
                    </button>
                  ))}
                </div>
              </motion.div>
            )}

            {messages.map((msg, i) => (
              <motion.div
                key={`${mode}-${i}`}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-3xl rounded-2xl px-5 py-4 ${
                    msg.role === "user"
                      ? "bg-indigo-500/20 border border-indigo-500/30 ml-4 md:ml-12"
                      : `glass mr-4 md:mr-12 ${msg.error ? "border-red-500/30" : ""}`
                  }`}
                >
                  {msg.role === "user" ? (
                    <p className="text-indigo-100">{msg.content}</p>
                  ) : (
                    <div className="space-y-4">
                      {msg.category && (
                        <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider bg-purple-500/15 text-purple-300 border border-purple-500/20">
                          {msg.category}
                        </span>
                      )}

                      {msg.content ? (
                        <div className="answer-stream">
                          <div className="markdown-body text-sm leading-relaxed text-gray-300">
                            <ReactMarkdown>{msg.content}</ReactMarkdown>
                          </div>
                        </div>
                      ) : (
                        <TypingIndicator />
                      )}

                      {msg.error && (
                        <button
                          onClick={() => handleRetry(i)}
                          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 transition-colors"
                        >
                          <RefreshCw size={12} /> Retry
                        </button>
                      )}

                      {msg.confidence && !msg.error && (
                        <div className="flex items-center gap-2 flex-wrap">
                          <ConfidenceBadge level={msg.confidence} />
                          <button
                            onClick={() => handleCopy(msg.content, i)}
                            className="p-1.5 rounded-lg hover:bg-white/5 text-gray-500 hover:text-gray-300 transition-colors"
                            title="Copy answer"
                          >
                            {copiedIdx === i ? <Check size={14} className="text-emerald-400" /> : <Copy size={14} />}
                          </button>
                          <button
                            onClick={() => {
                              const userQ = messages[i - 1]?.content || "Legal question";
                              handleExportPDF(userQ, msg.content);
                            }}
                            className="p-1.5 rounded-lg hover:bg-white/5 text-gray-500 hover:text-gray-300 transition-colors"
                            title="Export as PDF"
                          >
                            <Download size={14} />
                          </button>
                        </div>
                      )}

                      {msg.sources && msg.sources.length > 0 && (
                        <div className="pt-2 border-t border-indigo-500/10">
                          <p className="text-xs text-gray-500 mb-2 font-semibold uppercase tracking-wider">Sources</p>
                          <SourceChips sources={msg.sources} />
                        </div>
                      )}

                      {msg.chunks && msg.chunks.length > 0 && (
                        <div className="pt-2">
                          <button
                            onClick={() => setExpandedChunk(expandedChunk === i ? null : i)}
                            className="flex items-center gap-1 text-xs text-gray-500 hover:text-indigo-400 transition-colors"
                          >
                            {expandedChunk === i ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                            {expandedChunk === i ? "Hide" : "View"} retrieved context ({msg.chunks.length} chunks)
                          </button>
                          <AnimatePresence>
                            {expandedChunk === i && (
                              <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: "auto", opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                className="mt-2 space-y-2 overflow-hidden"
                              >
                                {msg.chunks.map((chunk, ci) => (
                                  <div key={ci} className="p-3 rounded-lg bg-white/[0.02] border border-indigo-500/10 text-xs">
                                    <div className="flex justify-between mb-1">
                                      <span className="text-indigo-400 font-medium">{chunk.source}</span>
                                      <span className="text-gray-600">score: {chunk.score.toFixed(4)}</span>
                                    </div>
                                    <p className="text-gray-500 line-clamp-3">{chunk.text}</p>
                                  </div>
                                ))}
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                      )}

                      {msg.relatedQuestions && msg.relatedQuestions.length > 0 && (
                        <div className="pt-2 border-t border-indigo-500/10">
                          <p className="text-xs text-gray-500 mb-2 font-semibold uppercase tracking-wider">Related Questions</p>
                          <div className="flex flex-wrap gap-2">
                            {msg.relatedQuestions.map((rq, ri) => (
                              <button
                                key={ri}
                                onClick={() => handleSubmit(rq)}
                                className="text-xs px-3 py-1.5 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 hover:bg-indigo-500/20 transition-colors text-left"
                              >
                                {rq.length > 80 ? rq.slice(0, 80) + "..." : rq}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </motion.div>
            ))}

            {isStreaming && messages[messages.length - 1]?.content === "" && <TypingIndicator />}
            <div ref={chatEndRef} />
          </div>

          {mode === "assistant" && messages.length > 0 && (
            <div className="mx-4 md:mx-8 mb-2 px-4 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-start gap-2">
              <AlertTriangle size={16} className="text-amber-400 mt-0.5 shrink-0" />
              <p className="text-xs text-amber-300/80">
                This is AI-generated legal information, not professional legal advice. Consult a qualified lawyer for your specific situation.
              </p>
            </div>
          )}

          <div className="p-3 md:p-4 px-4 md:px-8 border-t border-indigo-500/10 glass">
            <div className="flex items-end gap-2 md:gap-3 max-w-4xl mx-auto">
              <div className="flex-1 relative">
                <textarea
                  ref={inputRef}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit();
                    }
                  }}
                  placeholder={
                    mode === "search"
                      ? "Ask any legal question... (⌘K)"
                      : "Describe your legal problem... (⌘K)"
                  }
                  rows={1}
                  className="w-full bg-white/5 border border-indigo-500/20 rounded-xl px-4 py-3 pr-12 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-indigo-500/50 resize-none"
                  disabled={isStreaming}
                />
              </div>
              <button
                onClick={() => handleSubmit()}
                disabled={isStreaming || !query.trim()}
                className="shrink-0 w-11 h-11 rounded-xl bg-indigo-500 hover:bg-indigo-400 disabled:bg-indigo-500/30 disabled:cursor-not-allowed flex items-center justify-center transition-all shadow-lg shadow-indigo-500/25"
              >
                <Send size={18} className="text-white" />
              </button>
            </div>
          </div>
        </main>
      </div>
    </>
  );
}
