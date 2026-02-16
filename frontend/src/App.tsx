import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Search, Activity, BarChart3, Target, AlertCircle } from 'lucide-react';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, ComposedChart, RadarChart, PolarGrid,
    PolarAngleAxis, Radar
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

// --- Error Boundary ---
class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean }> {
    constructor(props: { children: React.ReactNode }) {
        super(props);
        this.state = { hasError: false };
    }
    static getDerivedStateFromError() { return { hasError: true }; }
    render() {
        if (this.state.hasError) {
            return (
                <div className="min-h-screen bg-black flex items-center justify-center p-6 text-center">
                    <div className="glass-panel p-10 max-w-lg border-red-500/20">
                        <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-6" />
                        <h1 className="text-2xl font-black text-white mb-4 uppercase tracking-tighter">System Critical Failure</h1>
                        <button onClick={() => window.location.reload()} className="bg-red-500 text-white px-8 py-3 rounded-2xl font-bold uppercase text-xs tracking-widest mt-6 hover:bg-red-600 transition-all">Re-Launch Dashboard</button>
                    </div>
                </div>
            );
        }
        return this.props.children;
    }
}

const API_BASE = "https://tryquant.onrender.com/"; // Use relative path since we're serving from the same port

function App() {
    const [searchInput, setSearchInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);
    const [timeframe, setTimeframe] = useState('1y');
    const [mcTimeframe, setMcTimeframe] = useState<7 | 15 | 30 | 60>(30);
    const [suggestions, setSuggestions] = useState<any[]>([]);
    const [showSuggestions, setShowSuggestions] = useState(false);

    const [scenarioActive, setScenarioActive] = useState(false);
    const [volatilityShock, setVolatilityShock] = useState(1.0);
    const [sentimentDrift, setSentimentDrift] = useState(0.0);

    const fetchData = useCallback(async (ticker: string, enableLoading: boolean = true) => {
        if (enableLoading) setLoading(true);
        setError(null);
        try {
            const response = await axios.get(`${API_BASE}/predict/${ticker}`, {
                params: {
                    days: mcTimeframe,
                    iterations: 1000,
                    vol_multiplier: scenarioActive ? volatilityShock : 1.0,
                    sentiment_bias: scenarioActive ? sentimentDrift : 0.0
                }
            });
            setData(response.data);
            setSuggestions([]);
            setShowSuggestions(false);
        } catch (err: any) {
            console.error('API Error Details:', {
                message: err.message,
                status: err.response?.status,
                data: err.response?.data
            });
            setError(err.response?.data?.detail || `System Error: ${err.message}. Ensure backend is running.`);
        } finally {
            if (enableLoading) setLoading(false);
        }
    }, [mcTimeframe, scenarioActive, volatilityShock, sentimentDrift]);

    // Debounced fetch for scenario slider changes
    useEffect(() => {
        if (!data?.ticker || !scenarioActive) return;

        const timeoutId = setTimeout(() => {
            fetchData(data.ticker, false); // Disable loading spinner for scenario updates
        }, 600); // 600ms debounce

        return () => clearTimeout(timeoutId);
    }, [volatilityShock, sentimentDrift, scenarioActive]);

    const handleScenarioToggle = () => {
        const newState = !scenarioActive;
        setScenarioActive(newState);
        // Immediate fetch when toggling
        if (data?.ticker) {
            // We need to pass the new state directly because the state update might be async/batched
            // But since fetchData uses the state from closure, we might need to be careful.
            // Actually, the useEffect above will handle the 'true' case (active).
            // For the 'false' case, we want to reset immediately.
            if (!newState) {
                // Resetting to defaults logically effectively, but we trigger a fetch with defaults logic in fetchData
                // We can force a fetch here, but we need to ensure fetchData reads the *latest* scenarioActive.
                // Since we just set it, it might not be updated in the closure yet if we call fetchData immediately.
                // However, we can trust the useEffect dependency on 'scenarioActive' to trigger the re-fetch.
            }
        }
    };

    useEffect(() => { fetchData('BTC-USD'); }, []);

    useEffect(() => {
        const fetchSuggestions = async () => {
            if (searchInput.length < 2) { setSuggestions([]); return; }
            try {
                const response = await axios.get(`${API_BASE}/search`, { params: { query: searchInput } });
                if (Array.isArray(response.data)) { setSuggestions(response.data); setShowSuggestions(true); } else { setSuggestions([]); }
            } catch (err) { console.error('Search error:', err); }
        };
        const timeoutId = setTimeout(fetchSuggestions, 300);
        return () => clearTimeout(timeoutId);
    }, [searchInput]);

    const handleSearch = (e: React.FormEvent) => { e.preventDefault(); if (searchInput) fetchData(searchInput.toUpperCase()); };
    const selectSuggestion = (symbol: string) => { setSearchInput(symbol); fetchData(symbol); setShowSuggestions(false); };

    const getSignalColor = (signal: string) => {
        switch (signal?.toUpperCase()) {
            case 'STRONG BUY': return 'text-green-400';
            case 'BUY': return 'text-emerald-500';
            case 'STRONG SELL': return 'text-red-500';
            case 'SELL': return 'text-orange-500';
            default: return 'text-gray-400';
        }
    };

    const getSignalBg = (signal: string) => {
        switch (signal?.toUpperCase()) {
            case 'STRONG BUY': return 'bg-green-500/10 border-green-500/20';
            case 'BUY': return 'bg-emerald-500/10 border-emerald-500/20';
            case 'STRONG SELL': return 'bg-red-500/10 border-red-500/20';
            case 'SELL': return 'bg-orange-500/10 border-orange-500/20';
            default: return 'bg-gray-500/10 border-gray-500/20';
        }
    };

    const getSimulationData = useMemo(() => {
        if (!data?.simulation_data) return [];
        return data.simulation_data.slice(0, mcTimeframe + 1).map((d: any) => ({
            ...d,
            range: [d.p10, d.p90]
        }));
    }, [data, mcTimeframe]);

    const getHistoricalData = useMemo(() => {
        if (!data?.historical_data) return [];
        const sliceCount = timeframe === '1y' ? 252 : timeframe === '6m' ? 126 : 63;
        return data.historical_data.slice(-sliceCount);
    }, [data, timeframe]);

    return (
        <ErrorBoundary>
            <div className="min-h-screen bg-[#020202] text-white font-sans selection:bg-purple-500/30 overflow-x-hidden">
                <div className="fixed inset-0 pointer-events-none opacity-30">
                    <div className="absolute inset-0 bg-[linear-gradient(rgba(18,18,18,0)_1px,transparent_1px),linear-gradient(90deg,rgba(18,18,18,0)_1px,transparent_1px)] bg-[size:32px_32px]" />
                </div>

                <AnimatePresence>
                    {loading && (
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-[100] bg-black/90 backdrop-blur-md flex flex-col items-center justify-center">
                            <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 2, ease: "linear" }} className="w-16 h-16 border-t-2 border-purple-500 rounded-full mb-4" />
                            <h2 className="text-sm font-bold uppercase tracking-[0.3em] text-purple-400">Synchronizing Market Intel</h2>
                        </motion.div>
                    )}
                </AnimatePresence>

                <div className="relative z-10 max-w-[1920px] mx-auto p-4 flex flex-col gap-4">

                    {/* --- HEADER & SEARCH --- */}
                    <div className="flex flex-col lg:flex-row items-center justify-between gap-4 py-2 border-b border-white/5">
                        <div className="flex items-center gap-4">
                            <div className="w-10 h-10 bg-gradient-to-tr from-purple-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/20">
                                <Activity className="w-5 h-5 text-white" />
                            </div>
                            <div>
                                <h1 className="text-3xl font-black uppercase tracking-tighter leading-none text-white">FusionQuant<span className="text-purple-500 italic">AI</span></h1>
                                <span className="text-xs text-gray-500 font-bold tracking-widest uppercase">Hybrid Intelligence Terminal</span>
                            </div>
                        </div>


                        <form onSubmit={handleSearch} className="relative group w-full lg:w-[600px]">
                            <input
                                type="text"
                                value={searchInput}
                                onChange={(e) => setSearchInput(e.target.value)}
                                placeholder="TICKER (e.g. AAPL, BTC-USD, TSLA)"
                                className="w-full bg-[#0a0a0a] border border-white/10 rounded-xl px-12 py-3 text-sm font-mono focus:border-purple-500/50 outline-none transition-all placeholder:text-gray-700 text-white"
                            />
                            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-700 group-focus-within:text-purple-500 transition-colors" />
                            <AnimatePresence>
                                {showSuggestions && suggestions.length > 0 && (
                                    <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} className="absolute top-full left-0 right-0 mt-2 bg-black border border-white/10 rounded-xl overflow-hidden z-[90] shadow-2xl">
                                        {suggestions.map((s, idx) => (
                                            <button key={idx} onClick={() => selectSuggestion(s.symbol)} className="w-full flex justify-between p-3 hover:bg-white/5 text-left text-xs border-b border-white/5 last:border-0 border-white/10">
                                                <span className="font-bold text-white border-r border-white/10 pr-3 mr-3 w-20">{s.symbol}</span>
                                                <span className="text-gray-500 truncate flex-1">{s.name}</span>
                                                <span className="text-[10px] text-gray-700 uppercase ml-2">{s.exchange}</span>
                                            </button>
                                        ))}
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </form>
                    </div>



                    {error && (
                        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl flex items-center gap-4">
                            <AlertCircle className="w-5 h-5 text-red-500" />
                            <p className="text-xs font-bold text-red-400 uppercase tracking-tighter">{error}</p>
                        </motion.div>
                    )}

                    {data && (
                        <div className="grid grid-cols-12 gap-4">

                            {/* --- ROW 1: CORE SIGNAL & METRICS --- */}
                            {/* --- ROW 1: COMMAND CENTER --- */}

                            {/* 1. Market Signal (Compact) */}
                            <div className={`col-span-12 lg:col-span-3 glass-panel p-5 border-l-4 ${getSignalBg(data.market_signal)} flex flex-col justify-between relative overflow-hidden group min-h-[180px]`}>
                                <div className="absolute inset-0 opacity-10 bg-gradient-to-br from-transparent to-white/5 pointer-events-none" />

                                <div className="flex justify-between items-start w-full relative z-10">
                                    <div className="flex items-center gap-2">
                                        <div className={`w-2 h-2 rounded-full animate-pulse ${data.confidence > 80 ? 'bg-green-500' : 'bg-yellow-500'}`} />
                                        <span className="text-xs font-black uppercase tracking-widest text-gray-500">Primary Signal</span>
                                    </div>
                                    <span className="text-xs font-black font-mono text-white px-2 py-0.5 bg-white/5 rounded border border-white/10 shadow-sm">{data.ticker}</span>
                                </div>

                                <div className="flex items-center justify-center my-2 relative z-10">
                                    <h2 className={`text-6xl font-black tracking-tighter uppercase drop-shadow-2xl ${getSignalColor(data.market_signal)}`}>{data.market_signal}</h2>
                                </div>

                                <div className="w-full relative z-10">
                                    <div className="flex justify-between items-end mb-1">
                                        <span className="text-[10px] font-bold text-gray-600 uppercase">Confidence</span>
                                        <span className="text-sm font-mono font-black text-white">{data.confidence?.toFixed(0)}%</span>
                                    </div>
                                    <div className="w-full h-1 bg-gray-800 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full transition-all duration-1000 ${data.confidence > 80 ? 'bg-green-500' : data.confidence > 50 ? 'bg-yellow-500' : 'bg-red-500'}`}
                                            style={{ width: `${data.confidence}%` }}
                                        />
                                    </div>
                                </div>
                                <div className="w-full relative z-10 mt-3 pt-2 border-t border-white/5">
                                    <div className="flex items-center gap-1 mb-1"><span className="text-[10px] font-black uppercase tracking-widest text-purple-400">Thesis AI Logic</span></div>
                                    <div className="flex flex-col gap-1">
                                        {data.signal_reasons?.slice(0, 2).map((r: string, i: number) => (
                                            <div key={i} className="flex items-start gap-1.5">
                                                <div className="w-1 h-1 mt-1 rounded-full bg-purple-500 flex-shrink-0" />
                                                <span className="text-[11px] font-bold text-gray-400 uppercase leading-none truncate">{r}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            {/* 2. Strategic Execution Setup (Compact) */}
                            <div className="col-span-12 lg:col-span-3 glass-panel p-5 bg-gradient-to-r from-purple-900/10 to-black relative flex flex-col justify-between min-h-[180px]">
                                <div className="flex justify-between items-start mb-2 border-b border-white/5 pb-2">
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs font-black uppercase tracking-widest text-purple-400 flex items-center gap-2"><Target className="w-3 h-3" /> Execution Setup</span>
                                    </div>
                                    <div className="flex gap-2">
                                        <span className="text-[10px] font-bold text-gray-500 uppercase">R:<span className="text-white">₹{data.institutional?.resistance?.toFixed(0)}</span></span>
                                        <span className="text-[10px] font-bold text-gray-500 uppercase">S:<span className="text-white">₹{data.institutional?.support?.toFixed(0)}</span></span>
                                    </div>
                                </div>

                                <div className="grid grid-cols-1 gap-1.5 py-1">
                                    <div className="flex justify-between items-baseline border-b border-white/5 pb-1">
                                        <span className="text-xs text-gray-500 font-bold uppercase">Entry Zone</span>
                                        <span className="text-2xl font-mono font-black text-white">₹{data.predicted_next_day_lstm?.toFixed(0)}</span>
                                    </div>
                                    <div className="flex justify-between items-baseline border-b border-white/5 pb-1">
                                        <span className="text-xs text-gray-500 font-bold uppercase">Current LTP</span>
                                        <span className="text-2xl font-mono font-black text-gray-300">₹{data.last_close?.toFixed(0)}</span>
                                    </div>
                                    <div className="flex justify-between items-center pt-1">
                                        <span className="text-xs text-gray-500 font-bold uppercase">Inst. Flow</span>
                                        <span className={`text-[11px] font-black px-2 py-0.5 rounded uppercase ${data.institutional?.flow_status === 'ACCUMULATION' ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>{data.institutional?.flow_status}</span>
                                    </div>
                                </div>

                                <div className="mt-2 flex items-center gap-2">
                                    <span className="text-[11px] text-gray-600 font-bold uppercase whitespace-nowrap">Trend Strength</span>
                                    <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden">
                                        <div className="h-full bg-purple-500" style={{ width: `${data.institutional?.adx || 0}%` }} />
                                    </div>
                                    <span className="text-[11px] font-mono text-purple-400">{data.institutional?.adx?.toFixed(0)}</span>
                                </div>
                            </div>

                            {/* 3. Technical Analysis Suite */}
                            <div className="col-span-12 lg:col-span-3 glass-panel p-5 bg-black/40 flex flex-col justify-between min-h-[180px] border-l-4 border-purple-500/20">
                                <div className="flex items-center gap-2 mb-2">
                                    <Activity className="w-3 h-3 text-purple-500" />
                                    <span className="text-xs font-black uppercase tracking-widest text-gray-500">Technical Analysis Suite</span>
                                </div>
                                <div className="space-y-2">
                                    <div className="flex justify-between border-b border-white/5 pb-1">
                                        <span className="text-xs text-gray-500 font-bold">RSI-14 <span className={`text-[10px] ml-1 ${data.rsi > 70 ? 'text-red-400' : data.rsi < 30 ? 'text-green-400' : 'text-gray-600'}`}>({data.rsi > 70 ? 'OVERBOUGHT' : data.rsi < 30 ? 'OVERSOLD' : 'NEUTRAL'})</span></span>
                                        <span className={`text-sm font-mono font-black ${data.rsi > 70 ? 'text-red-400' : data.rsi < 30 ? 'text-green-400' : 'text-purple-400'}`}>{data.rsi?.toFixed(1)}</span>
                                    </div>
                                    <div className="flex justify-between border-b border-white/5 pb-1"><span className="text-xs text-gray-500 font-bold">MACD</span><span className={`text-sm font-mono font-black ${data.macd > 0 ? 'text-green-400' : 'text-red-400'}`}>{data.macd?.toFixed(2)}</span></div>
                                    <div className="flex justify-between border-b border-white/5 pb-1"><span className="text-xs text-gray-500 font-bold">ADX (Trend)</span><span className="text-sm font-mono font-black text-blue-400">{data.institutional?.adx?.toFixed(1)}</span></div>
                                    <div className="flex justify-between border-b border-white/5 pb-1"><span className="text-xs text-gray-500 font-bold">BB Width</span><span className={`text-sm font-mono font-black ${data.institutional?.bb_width < 5 ? 'text-yellow-400 animate-pulse' : 'text-gray-300'}`}>{data.institutional?.bb_width?.toFixed(1)}%</span></div>
                                    <div className="flex justify-between border-b border-white/5 pb-1"><span className="text-xs text-gray-500 font-bold">OBV (5D)</span><span className={`text-sm font-mono font-black ${data.institutional?.obv_change > 0 ? 'text-green-400' : 'text-red-400'}`}>{data.institutional?.obv_change?.toFixed(1)}%</span></div>
                                </div>
                            </div>

                            {/* 4. Market Structure Radar (Moved to Row 1) */}
                            <div className="col-span-12 lg:col-span-3 glass-panel p-4 bg-black/40 min-h-[160px] border-white/5 relative">
                                <span className="absolute top-4 left-4 text-[9px] font-black uppercase text-gray-500 tracking-widest">Market Structure Radar</span>
                                <ResponsiveContainer width="100%" height="100%">
                                    <RadarChart outerRadius="70%" data={[{ x: 'Momentum', v: data.radar?.momentum }, { x: 'Trend', v: data.radar?.trend }, { x: 'Volatility', v: data.radar?.volatility }, { x: 'Strength', v: data.radar?.strength }, { x: 'Inst. Flow', v: (data.institutional?.breakout_prob || 50) }]}>
                                        <PolarGrid stroke="#222" />
                                        <PolarAngleAxis dataKey="x" tick={{ fill: '#444', fontSize: 8, fontWeight: 'bold' }} />
                                        <Radar dataKey="v" stroke="#a855f7" fill="#a855f7" fillOpacity={0.3} />
                                    </RadarChart>
                                </ResponsiveContainer>
                            </div>


                            {/* --- ROW 2: MARKET VISION & DEEP DIVE --- */}

                            {/* Market Overview (Left Column) */}
                            <div className="col-span-12 lg:col-span-3 flex flex-col gap-4">
                                <div className="glass-panel p-4 flex flex-col h-[340px] bg-black/40 border-white/5 relative overflow-hidden">
                                    <div className="flex items-center gap-2 mb-4 z-10">
                                        <Activity className="w-4 h-4 text-orange-500" />
                                        <span className="text-xs font-black uppercase tracking-widest text-gray-500">Overall Market Behavior</span>
                                    </div>

                                    {/* Market Mood Indicator */}
                                    <div className="mb-4 bg-white/5 rounded-xl p-3 border border-white/5 relative overflow-hidden z-10">
                                        <div className="absolute inset-0 bg-gradient-to-r from-orange-500/10 to-transparent opacity-50" />
                                        <span className="text-[11px] font-bold text-gray-400 uppercase tracking-wider block mb-1">Market Sentiment</span>
                                        <div className="flex items-center justify-between">
                                            <span className={`text-lg font-black uppercase tracking-tighter ${data.market_mood?.includes('Bullish') ? 'text-green-400' :
                                                data.market_mood?.includes('Bearish') ? 'text-red-400' :
                                                    data.market_mood?.includes('Fear') ? 'text-orange-400' : 'text-gray-200'
                                                }`}>
                                                {data.market_mood || 'NEUTRAL'}
                                            </span>
                                            {data.market_overview?.find((m: any) => m.name === 'VIX') && (
                                                <div className="text-[11px] font-mono text-gray-500">
                                                    VIX: <span className="text-white font-bold">{data.market_overview.find((m: any) => m.name === 'VIX').price.toFixed(2)}</span>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    <div className="space-y-2 z-10 flex-1 overflow-y-auto pr-1 custom-scrollbar">
                                        {data.market_overview && data.market_overview.length > 0 ? (
                                            data.market_overview.filter((m: any) => m.name !== 'VIX').map((m: any, idx: number) => (
                                                <div key={idx} className="flex justify-between items-center bg-white/5 p-2 rounded border border-white/5 hover:border-white/10 transition-colors">
                                                    <div>
                                                        <p className="text-[11px] font-bold text-gray-400 uppercase">{m.name}</p>
                                                        <p className="text-sm font-mono font-black text-white">
                                                            {m.price > 1000 ? `₹${m.price.toLocaleString()}` : `$${m.price.toFixed(2)}`}
                                                        </p>
                                                    </div>
                                                    <div className={`text-[11px] font-black px-1.5 py-0.5 rounded ${m.change >= 0 ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
                                                        {m.change >= 0 ? '+' : ''}{m.change.toFixed(2)}%
                                                    </div>
                                                </div>
                                            ))
                                        ) : (
                                            <div className="text-center text-[10px] text-gray-600 italic py-4">Market data unavailable</div>
                                        )}
                                    </div>
                                    {/* Decorative background element */}
                                    <div className="absolute -bottom-10 -right-10 w-32 h-32 bg-orange-500/5 rounded-full blur-2xl pointer-events-none" />
                                </div>

                                {/* Scenario Analysis (Moved back here) */}
                                <div className={`glass-panel p-4 flex flex-col transition-all duration-300 ${scenarioActive ? 'border-purple-500/50 shadow-[0_0_30px_rgba(168,85,247,0.15)] bg-purple-900/10' : 'bg-black/40 border-white/5 opacity-80 hover:opacity-100'}`}>
                                    <div className="flex items-center justify-between mb-4">
                                        <div className="flex items-center gap-2">
                                            <Target className={`w-4 h-4 ${scenarioActive ? 'text-purple-400' : 'text-gray-500'}`} />
                                            <span className={`text-xs font-black uppercase tracking-widest ${scenarioActive ? 'text-white' : 'text-gray-500'}`}>Scenario Simulation</span>
                                        </div>
                                        <button
                                            onClick={handleScenarioToggle}
                                            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-black ${scenarioActive ? 'bg-purple-600' : 'bg-gray-700'}`}
                                        >
                                            <span className={`${scenarioActive ? 'translate-x-5' : 'translate-x-1'} inline-block h-3 w-3 transform rounded-full bg-white transition-transform`} />
                                        </button>
                                    </div>

                                    <div className={`space-y-4 transition-all duration-500 ${scenarioActive ? 'opacity-100 filter-none' : 'opacity-50 blur-[1px] pointer-events-none'}`}>
                                        {/* Volatility Slider */}
                                        <div>
                                            <div className="flex justify-between mb-1">
                                                <span className="text-xs font-bold text-gray-400 uppercase">Volatility Shock</span>
                                                <span className="text-xs font-mono font-black text-purple-400">{volatilityShock}x</span>
                                            </div>
                                            <input
                                                type="range"
                                                min="0.5"
                                                max="3.0"
                                                step="0.1"
                                                value={volatilityShock}
                                                onChange={(e) => setVolatilityShock(parseFloat(e.target.value))}
                                                className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500 hover:accent-purple-400"
                                            />
                                            <div className="flex justify-between mt-1 text-[10px] text-gray-600 font-mono">
                                                <span>Stable</span>
                                                <span>Crisis</span>
                                            </div>
                                        </div>

                                        {/* Sentiment Slider */}
                                        <div>
                                            <div className="flex justify-between mb-1">
                                                <span className="text-xs font-bold text-gray-400 uppercase">Sentiment Bias</span>
                                                <span className={`text-xs font-mono font-black ${sentimentDrift > 0 ? 'text-green-400' : sentimentDrift < 0 ? 'text-red-400' : 'text-gray-400'}`}>
                                                    {sentimentDrift > 0 ? '+' : ''}{sentimentDrift}
                                                </span>
                                            </div>
                                            <input
                                                type="range"
                                                min="-0.5"
                                                max="0.5"
                                                step="0.05"
                                                value={sentimentDrift}
                                                onChange={(e) => setSentimentDrift(parseFloat(e.target.value))}
                                                className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500 hover:accent-purple-400"
                                            />
                                            <div className="flex justify-between mt-1 text-[10px] text-gray-600 font-mono">
                                                <span>Bearish</span>
                                                <span>Bullish</span>
                                            </div>
                                        </div>
                                    </div>

                                    {scenarioActive && (
                                        <div className="mt-3 pt-3 border-t border-purple-500/20 text-center">
                                            <span className="text-[10px] text-purple-300 font-bold uppercase tracking-wider animate-pulse">Running Custom Simulation...</span>
                                        </div>
                                    )}
                                </div>
                            </div>


                            {/* Charts Area (Right Column) */}
                            <div className="col-span-12 lg:col-span-9 flex flex-col gap-4">
                                <div className="glass-panel p-4 bg-black/20 h-[300px] flex flex-col relative overflow-hidden">
                                    <div className="absolute top-0 right-0 p-4 opacity-10 pointer-events-none"><BarChart3 className="w-64 h-64 -rotate-12" /></div>
                                    <div className="flex justify-between items-center mb-4 z-10">
                                        <div className="flex items-center gap-2"><BarChart3 className="w-4 h-4 text-purple-500" /><span className="text-xs font-black uppercase tracking-widest text-gray-500">Price Action & Momentum</span></div>
                                        <div className="flex bg-white/5 rounded-lg p-0.5 border border-white/5">{['1d', '5d', '1w', '1m', '6m', '1y', '5y'].map((tf) => <button key={tf} onClick={() => setTimeframe(tf)} className={`px-2 py-0.5 text-[10px] font-bold rounded-md transition-all ${timeframe === tf ? 'bg-purple-600 text-white shadow-lg' : 'text-gray-500 hover:text-white'}`}>{tf.toUpperCase()}</button>)}</div>
                                    </div>
                                    <div className="flex-1 w-full min-h-0 z-10">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <ComposedChart data={getHistoricalData}>
                                                <defs>
                                                    <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#a855f7" stopOpacity={0.3} /><stop offset="95%" stopColor="#a855f7" stopOpacity={0} /></linearGradient>
                                                </defs>
                                                <CartesianGrid stroke="#111" vertical={false} />
                                                <XAxis dataKey="Date" hide />
                                                <YAxis domain={['auto', 'auto']} stroke="#333" fontSize={12} axisLine={false} tickLine={false} orientation="right" />
                                                <Tooltip contentStyle={{ backgroundColor: '#050505', border: '1px solid #222', borderRadius: '8px' }} itemStyle={{ fontSize: '12px', fontWeight: 'bold' }} />
                                                <Area type="monotone" dataKey="Close" stroke="#a855f7" strokeWidth={2} fillOpacity={1} fill="url(#colorClose)" />
                                            </ComposedChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>



                                {/* Scenario Analysis Toolbar (Moved Here) */}
                                <div className="glass-panel p-4 bg-black/20 h-[300px] flex flex-col border-emerald-500/10 relative">
                                    <div className="flex justify-between items-center mb-4">
                                        <div className="flex items-center gap-2"><Activity className="w-4 h-4 text-emerald-500" /><span className="text-xs font-black uppercase tracking-widest text-gray-500">Predictive Neural Projection</span></div>
                                        <div className="flex bg-white/5 rounded-lg p-0.5 border border-white/5">{[7, 15, 30, 60].map((d) => <button key={d} onClick={() => setMcTimeframe(d as any)} className={`px-3 py-0.5 text-[10px] font-bold rounded-md transition-all ${mcTimeframe === d ? 'bg-emerald-600 text-white shadow-lg' : 'text-gray-500 hover:text-white'}`}>{d}D</button>)}</div>
                                    </div>
                                    <div className="flex-1 w-full min-h-0">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <AreaChart data={getSimulationData}>
                                                <CartesianGrid stroke="#111" vertical={false} />
                                                <XAxis dataKey="day" stroke="#333" fontSize={12} axisLine={false} tickLine={false} />
                                                <YAxis domain={['auto', 'auto']} stroke="#333" fontSize={12} axisLine={false} tickLine={false} orientation="right" />
                                                <Tooltip contentStyle={{ backgroundColor: '#050505', border: '1px solid #222', borderRadius: '8px' }} />
                                                <Area dataKey="range" stroke="none" fill="#10b981" fillOpacity={0.1} animationDuration={1000} />
                                                <Area type="monotone" dataKey="mean" stroke="#10b981" strokeWidth={2} fill="none" dot={false} />
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            </div>



                        </div>
                    )}

                    {!data && !loading && !error && (
                        <div className="flex flex-col items-center justify-center h-[70vh] text-center opacity-40">
                            <motion.div animate={{ y: [0, -10, 0] }} transition={{ repeat: Infinity, duration: 3 }}><Activity className="w-20 h-20 text-gray-500 mb-6" /></motion.div>
                            <h2 className="text-2xl font-black uppercase tracking-[0.4em] text-gray-600 mb-2">Neural Link Idle</h2>
                            <p className="text-sm font-mono text-gray-700">AWAITING SYSTEM COMMAND OR TICKER INPUT...</p>
                        </div>
                    )}
                </div>
            </div>
        </ErrorBoundary >
    );
}

export default App;
