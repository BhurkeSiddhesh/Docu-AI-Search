import React, { useState } from 'react';
import { Settings, Moon, Sun, Cpu, ChevronDown, Check, History, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const Header = ({ darkMode, toggleDarkMode, openSettings, toggleHistory, isHistoryOpen, activeModel, availableModels = [], onModelChange, indexingStatus }) => {
    const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);

    const handleModelSelect = (modelName) => {
        onModelChange(modelName);
        setIsModelDropdownOpen(false);
    };

    return (
        <header className="sticky top-4 z-40 w-[calc(100%-2rem)] mx-auto liquid-glass px-2">
            <div className="flex h-16 items-center px-4">
                {/* Left: History + Logo */}
                <div className="flex items-center gap-3">
                    <button
                        onClick={toggleHistory}
                        className={`p-2.5 rounded-full transition-all duration-300 hover:bg-liquid-cobalt/10 hover:text-liquid-cobalt ${isHistoryOpen ? 'bg-liquid-cobalt/15 text-liquid-cobalt' : 'text-on-surface-variant'}`}
                        aria-label="Toggle history"
                    >
                        <History className="w-5 h-5" />
                    </button>
                    <div className="flex items-center gap-2">
                        <div className="w-9 h-9 rounded-full bg-gradient-to-br from-liquid-cobalt to-[#2e5bff] flex items-center justify-center shadow-lg shadow-liquid-cobalt/20">
                            <Sparkles className="w-4.5 h-4.5 text-white" />
                        </div>
                        <h1 className="text-xl font-black tracking-tighter text-on-surface font-display">
                            Docu<span className="text-liquid-cobalt">AI</span>
                        </h1>
                    </div>
                </div>

                {/* Center: Model Selector */}
                <div className="flex-1 flex items-center justify-center">
                    {activeModel && (
                        <div className="relative">
                            <button
                                onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
                                className="flex items-center gap-2.5 px-5 py-2.5 rounded-full surface-low hover:surface-lowest transition-all text-sm font-semibold group shadow-sm"
                            >
                                <div className="p-1.5 rounded-full bg-liquid-cobalt/10">
                                    <Cpu className="w-3.5 h-3.5 text-liquid-cobalt" />
                                </div>
                                <span className="truncate max-w-[120px] sm:max-w-[200px] text-on-surface">{activeModel}</span>
                                <ChevronDown className={`w-4 h-4 text-on-surface-variant transition-transform duration-300 ${isModelDropdownOpen ? 'rotate-180' : ''}`} />
                            </button>

                            <AnimatePresence>
                                {isModelDropdownOpen && (
                                    <motion.div
                                        initial={{ opacity: 0, y: -8, scale: 0.95 }}
                                        animate={{ opacity: 1, y: 8, scale: 1 }}
                                        exit={{ opacity: 0, y: -8, scale: 0.95 }}
                                        transition={{ duration: 0.2 }}
                                        className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-72 p-2 rounded-xl surface-lowest shadow-2xl border-none"
                                    >
                                        <div className="text-xs font-bold text-muted-foreground uppercase tracking-wider px-3 py-2">
                                            Available Models
                                        </div>
                                        <div className="max-h-64 overflow-y-auto space-y-1">
                                            {availableModels.length > 0 ? (
                                                availableModels.map((model) => {
                                                    const modelName = model.name.replace('.gguf', '');
                                                    const isActive = activeModel === modelName;
                                                    return (
                                                        <button
                                                            key={model.name}
                                                            onClick={() => handleModelSelect(modelName)}
                                                            className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-all flex items-center justify-between group
                                                                ${isActive
                                                                    ? 'bg-liquid-cobalt/10 text-liquid-cobalt font-bold'
                                                                    : 'hover:bg-liquid-surface text-on-surface-variant hover:text-on-surface'}`}
                                                        >
                                                            <span className="truncate">{modelName}</span>
                                                            {isActive && (
                                                                <div className="p-1 rounded-full bg-liquid-cobalt/20">
                                                                    <Check className="w-3 h-3 text-liquid-cobalt" />
                                                                </div>
                                                            )}
                                                        </button>
                                                    );
                                                })
                                            ) : (
                                                <div className="px-3 py-4 text-sm text-muted-foreground text-center">
                                                    No local models found
                                                    <p className="text-xs mt-1">Go to Settings → Model Manager</p>
                                                </div>
                                            )}
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    )}

                    {/* Indexing Indicator */}
                    {indexingStatus?.running && (
                        <div className="ml-4 flex items-center gap-2 px-3 py-1.5 rounded-full bg-liquid-cobalt/10">
                            <div className="w-2 h-2 rounded-full bg-liquid-cobalt animate-pulse" />
                            <span className="text-xs font-bold text-liquid-cobalt">{indexingStatus.progress}%</span>
                        </div>
                    )}
                </div>

                {/* Right: Theme + Settings */}
                <div className="flex items-center gap-2">
                    <button
                        onClick={toggleDarkMode}
                        className="p-2.5 rounded-full transition-all duration-300 hover:bg-liquid-surface text-on-surface-variant hover:text-on-surface"
                        aria-label="Toggle theme"
                    >
                        {darkMode ? (
                            <Sun className="w-5 h-5" />
                        ) : (
                            <Moon className="w-5 h-5" />
                        )}
                    </button>
                    <button
                        onClick={openSettings}
                        className="p-2.5 rounded-full transition-all duration-300 hover:bg-liquid-surface text-on-surface-variant hover:text-on-surface"
                        aria-label="Settings"
                    >
                        <Settings className="w-5 h-5" />
                    </button>
                </div>
            </div>
        </header>
    );
};

export default Header;
