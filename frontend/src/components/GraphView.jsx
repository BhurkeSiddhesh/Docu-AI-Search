import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { RefreshCw, FileText, Hash, ZoomIn, ZoomOut, Maximize2, Network } from 'lucide-react';
import api from '../lib/api';
import logger from '../lib/logger';

const TYPE_COLORS = {
    '.pdf':  '#f43f5e', // rose-500
    '.docx': '#3b82f6', // blue-500
    '.xlsx': '#10b981', // emerald-500
    '.csv':  '#14b8a6', // teal-500
    '.pptx': '#f97316', // orange-500
    '.txt':  '#8b5cf6', // violet-500
    '.md':   '#a855f7', // purple-500
};
const DOC_FALLBACK = '#6366f1'; // indigo-500
const KEYWORD_COLOR = '#94a3b8'; // slate-400

const WIDTH = 1200;
const HEIGHT = 800;

/**
 * Deterministic force-directed layout (repulsion + edge springs + centering).
 * Runs synchronously — fine for the few hundred nodes a local library produces.
 */
function computeLayout(nodes, edges) {
    const pos = new Map();
    const n = nodes.length;
    if (!n) return pos;

    // Seed on a spiral so the simulation starts stable and is deterministic
    nodes.forEach((node, i) => {
        const angle = i * 2.399963; // golden angle
        const radius = 40 + 14 * Math.sqrt(i);
        pos.set(node.id, {
            x: WIDTH / 2 + radius * Math.cos(angle),
            y: HEIGHT / 2 + radius * Math.sin(angle),
            vx: 0,
            vy: 0,
        });
    });

    const springs = edges
        .filter((e) => pos.has(e.source_id) && pos.has(e.target_id))
        .map((e) => ({
            a: e.source_id,
            b: e.target_id,
            length: e.relation_type === 'similar_to' ? 170 : 110,
            strength: e.relation_type === 'similar_to' ? 0.04 : 0.025,
        }));

    const ids = nodes.map((d) => d.id);
    const REPULSION = 9000;
    const CENTER_PULL = 0.004;
    const DAMPING = 0.82;
    const ITERATIONS = 280;

    for (let iter = 0; iter < ITERATIONS; iter++) {
        for (let i = 0; i < ids.length; i++) {
            const p1 = pos.get(ids[i]);
            for (let j = i + 1; j < ids.length; j++) {
                const p2 = pos.get(ids[j]);
                let dx = p1.x - p2.x;
                let dy = p1.y - p2.y;
                let distSq = dx * dx + dy * dy;
                if (distSq < 1) { dx = (Math.random() - 0.5); dy = (Math.random() - 0.5); distSq = 1; }
                const force = REPULSION / distSq;
                const dist = Math.sqrt(distSq);
                const fx = (dx / dist) * force;
                const fy = (dy / dist) * force;
                p1.vx += fx; p1.vy += fy;
                p2.vx -= fx; p2.vy -= fy;
            }
        }

        for (const s of springs) {
            const p1 = pos.get(s.a);
            const p2 = pos.get(s.b);
            const dx = p2.x - p1.x;
            const dy = p2.y - p1.y;
            const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
            const force = (dist - s.length) * s.strength;
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;
            p1.vx += fx; p1.vy += fy;
            p2.vx -= fx; p2.vy -= fy;
        }

        for (const id of ids) {
            const p = pos.get(id);
            p.vx += (WIDTH / 2 - p.x) * CENTER_PULL;
            p.vy += (HEIGHT / 2 - p.y) * CENTER_PULL;
            p.x += p.vx * DAMPING;
            p.y += p.vy * DAMPING;
            p.vx *= DAMPING;
            p.vy *= DAMPING;
        }
    }

    // Fit the settled layout to ~85% of the canvas regardless of graph size
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of pos.values()) {
        minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
        minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    }
    const spanX = Math.max(maxX - minX, 1);
    const spanY = Math.max(maxY - minY, 1);
    const scale = Math.min((WIDTH * 0.85) / spanX, (HEIGHT * 0.82) / spanY);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    for (const p of pos.values()) {
        p.x = WIDTH / 2 + (p.x - cx) * scale;
        p.y = HEIGHT / 2 + (p.y - cy) * scale;
    }
    return pos;
}

/** Constellation-style empty state artwork. */
function EmptyGraphArt() {
    return (
        <svg viewBox="0 0 480 280" className="w-full max-w-md mx-auto" aria-hidden="true">
            <defs>
                <linearGradient id="eg-doc" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#6366f1" />
                    <stop offset="100%" stopColor="#8b5cf6" />
                </linearGradient>
                <radialGradient id="eg-glow" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#6366f1" stopOpacity="0.18" />
                    <stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
                </radialGradient>
            </defs>
            <circle cx="240" cy="140" r="130" fill="url(#eg-glow)" />
            {/* constellation lines */}
            <g stroke="#94a3b8" strokeOpacity="0.45" strokeWidth="1.5" strokeDasharray="3 4">
                <line x1="240" y1="130" x2="120" y2="70" />
                <line x1="240" y1="130" x2="360" y2="64" />
                <line x1="240" y1="130" x2="140" y2="212" />
                <line x1="240" y1="130" x2="352" y2="206" />
                <line x1="120" y1="70" x2="60" y2="140" />
                <line x1="360" y1="64" x2="412" y2="140" />
            </g>
            {/* satellite nodes */}
            <g>
                <circle cx="120" cy="70" r="9" fill="#f43f5e" opacity="0.85" />
                <circle cx="360" cy="64" r="9" fill="#10b981" opacity="0.85" />
                <circle cx="140" cy="212" r="9" fill="#f97316" opacity="0.85" />
                <circle cx="352" cy="206" r="9" fill="#3b82f6" opacity="0.85" />
                <circle cx="60" cy="140" r="5" fill="#94a3b8" opacity="0.7" />
                <circle cx="412" cy="140" r="5" fill="#94a3b8" opacity="0.7" />
            </g>
            {/* central document */}
            <g transform="translate(214 100)">
                <rect width="52" height="64" rx="8" fill="url(#eg-doc)" />
                <rect x="10" y="14" width="32" height="4" rx="2" fill="white" opacity="0.85" />
                <rect x="10" y="24" width="24" height="4" rx="2" fill="white" opacity="0.6" />
                <rect x="10" y="34" width="28" height="4" rx="2" fill="white" opacity="0.6" />
                <rect x="10" y="44" width="18" height="4" rx="2" fill="white" opacity="0.4" />
            </g>
        </svg>
    );
}

export default function GraphView() {
    const [graph, setGraph] = useState({ nodes: [], edges: [] });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [hovered, setHovered] = useState(null);
    const [zoom, setZoom] = useState(1);
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [dragNode, setDragNode] = useState(null);
    const [positions, setPositions] = useState(new Map());
    const svgRef = useRef(null);
    const panRef = useRef(null);

    const load = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await api.getGraph();
            const nodes = res.data?.nodes || [];
            const edges = res.data?.edges || [];
            setGraph({ nodes, edges });
            setPositions(computeLayout(nodes, edges));
        } catch (e) {
            logger.error('Failed to load knowledge graph', e);
            setError('Could not load the knowledge graph.');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { load(); }, [load]);

    const neighborMap = useMemo(() => {
        const map = new Map();
        for (const e of graph.edges) {
            if (!map.has(e.source_id)) map.set(e.source_id, new Set());
            if (!map.has(e.target_id)) map.set(e.target_id, new Set());
            map.get(e.source_id).add(e.target_id);
            map.get(e.target_id).add(e.source_id);
        }
        return map;
    }, [graph.edges]);

    const docCount = useMemo(() => graph.nodes.filter((n) => n.type === 'document').length, [graph.nodes]);
    const kwCount = graph.nodes.length - docCount;

    const isDimmed = (id) => {
        if (!hovered) return false;
        if (id === hovered) return false;
        return !(neighborMap.get(hovered)?.has(id));
    };

    const svgPoint = (evt) => {
        const svg = svgRef.current;
        if (!svg) return { x: 0, y: 0 };
        const rect = svg.getBoundingClientRect();
        return {
            x: ((evt.clientX - rect.left) / rect.width) * WIDTH / zoom - pan.x,
            y: ((evt.clientY - rect.top) / rect.height) * HEIGHT / zoom - pan.y,
        };
    };

    const onPointerMove = (evt) => {
        if (dragNode) {
            const pt = svgPoint(evt);
            setPositions((prev) => {
                const next = new Map(prev);
                const p = next.get(dragNode);
                if (p) next.set(dragNode, { ...p, x: pt.x, y: pt.y });
                return next;
            });
        } else if (panRef.current) {
            const { startX, startY, panX, panY } = panRef.current;
            setPan({
                x: panX + (evt.clientX - startX) / zoom,
                y: panY + (evt.clientY - startY) / zoom,
            });
        }
    };

    const stopInteractions = () => {
        setDragNode(null);
        panRef.current = null;
    };

    const nodeRadius = (node) => {
        if (node.type !== 'document') return 6;
        try {
            const meta = JSON.parse(node.metadata || '{}');
            return Math.min(14 + (meta.chunks || 1) * 1.5, 26);
        } catch {
            return 14;
        }
    };

    const nodeColor = (node) => {
        if (node.type !== 'document') return KEYWORD_COLOR;
        try {
            const meta = JSON.parse(node.metadata || '{}');
            return TYPE_COLORS[meta.file_type] || DOC_FALLBACK;
        } catch {
            return DOC_FALLBACK;
        }
    };

    if (loading) {
        return (
            <div className="px-4 sm:px-6 py-10 max-w-5xl mx-auto">
                <div className="card p-12 flex flex-col items-center gap-4">
                    <div className="flex gap-1">
                        <span className="typing-dot w-2 h-2 bg-primary rounded-full inline-block" />
                        <span className="typing-dot w-2 h-2 bg-primary rounded-full inline-block" />
                        <span className="typing-dot w-2 h-2 bg-primary rounded-full inline-block" />
                    </div>
                    <span className="text-sm text-slate-500 dark:text-slate-400">Mapping your knowledge…</span>
                </div>
            </div>
        );
    }

    if (error || graph.nodes.length === 0) {
        return (
            <div className="px-4 sm:px-6 py-10 max-w-5xl mx-auto animate-fade-in">
                <div className="card p-10 text-center">
                    <EmptyGraphArt />
                    <h2 className="mt-6 text-lg font-semibold text-slate-900 dark:text-slate-50">
                        {error ? 'Graph unavailable' : 'No knowledge graph yet'}
                    </h2>
                    <p className="mt-2 text-sm text-slate-500 dark:text-slate-400 max-w-sm mx-auto">
                        {error || 'Index a folder and Docu AI will map how your documents and their key topics connect.'}
                    </p>
                    <button onClick={load} className="btn-secondary mt-6">
                        <RefreshCw className="w-4 h-4" />
                        Refresh
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="px-4 sm:px-6 py-6 max-w-7xl mx-auto animate-fade-in">
            {/* Header */}
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg bg-primary text-white flex items-center justify-center">
                        <Network className="w-5 h-5" />
                    </div>
                    <div>
                        <h1 className="font-semibold text-slate-900 dark:text-slate-50">Knowledge graph</h1>
                        <p className="text-xs text-slate-500 dark:text-slate-400">
                            {docCount} documents · {kwCount} topics · {graph.edges.length} connections
                        </p>
                    </div>
                </div>
                <div className="flex items-center gap-1.5">
                    <button onClick={() => setZoom((z) => Math.min(z * 1.25, 4))} className="btn-ghost p-2" aria-label="Zoom in">
                        <ZoomIn className="w-4 h-4" />
                    </button>
                    <button onClick={() => setZoom((z) => Math.max(z / 1.25, 0.4))} className="btn-ghost p-2" aria-label="Zoom out">
                        <ZoomOut className="w-4 h-4" />
                    </button>
                    <button onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }} className="btn-ghost p-2" aria-label="Reset view">
                        <Maximize2 className="w-4 h-4" />
                    </button>
                    <button onClick={load} className="btn-secondary ml-1">
                        <RefreshCw className="w-4 h-4" />
                        Refresh
                    </button>
                </div>
            </div>

            {/* Canvas */}
            <div className="card overflow-hidden relative">
                <svg
                    ref={svgRef}
                    viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
                    className="w-full select-none touch-none"
                    style={{ aspectRatio: `${WIDTH} / ${HEIGHT}`, cursor: dragNode ? 'grabbing' : 'grab' }}
                    onPointerMove={onPointerMove}
                    onPointerUp={stopInteractions}
                    onPointerLeave={stopInteractions}
                    onPointerDown={(evt) => {
                        if (evt.target === svgRef.current || evt.target.dataset?.bg) {
                            panRef.current = { startX: evt.clientX, startY: evt.clientY, panX: pan.x, panY: pan.y };
                        }
                    }}
                >
                    <defs>
                        <radialGradient id="graph-bg-glow" cx="50%" cy="42%" r="60%">
                            <stop offset="0%" stopColor="#6366f1" stopOpacity="0.07" />
                            <stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
                        </radialGradient>
                    </defs>
                    <rect data-bg="1" width={WIDTH} height={HEIGHT} fill="url(#graph-bg-glow)" />

                    <g transform={`scale(${zoom}) translate(${pan.x} ${pan.y})`}>
                        {/* Edges */}
                        {graph.edges.map((e, i) => {
                            const p1 = positions.get(e.source_id);
                            const p2 = positions.get(e.target_id);
                            if (!p1 || !p2) return null;
                            const similar = e.relation_type === 'similar_to';
                            const dim = hovered && !(e.source_id === hovered || e.target_id === hovered);
                            return (
                                <line
                                    key={i}
                                    x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y}
                                    stroke={similar ? '#6366f1' : '#94a3b8'}
                                    strokeWidth={similar ? Math.max(1.5, e.weight * 3) : 1}
                                    strokeOpacity={dim ? 0.06 : similar ? 0.55 : 0.25}
                                    strokeDasharray={similar ? undefined : '2 3'}
                                />
                            );
                        })}

                        {/* Nodes */}
                        {graph.nodes.map((node) => {
                            const p = positions.get(node.id);
                            if (!p) return null;
                            const r = nodeRadius(node);
                            const color = nodeColor(node);
                            const dim = isDimmed(node.id);
                            const isDoc = node.type === 'document';
                            return (
                                <g
                                    key={node.id}
                                    transform={`translate(${p.x} ${p.y})`}
                                    opacity={dim ? 0.15 : 1}
                                    style={{ cursor: 'pointer', transition: 'opacity 150ms' }}
                                    onPointerEnter={() => setHovered(node.id)}
                                    onPointerLeave={() => setHovered(null)}
                                    onPointerDown={(evt) => { evt.stopPropagation(); setDragNode(node.id); }}
                                >
                                    {isDoc && <circle r={r + 5} fill={color} opacity="0.15" />}
                                    <circle
                                        r={r}
                                        fill={isDoc ? color : 'white'}
                                        stroke={color}
                                        strokeWidth={isDoc ? 0 : 2}
                                        className={isDoc ? '' : 'dark:fill-slate-900'}
                                    />
                                    {isDoc && (
                                        <FileText
                                            x={-r * 0.5} y={-r * 0.5}
                                            width={r} height={r}
                                            color="white"
                                            strokeWidth={2.2}
                                        />
                                    )}
                                    <text
                                        y={r + (isDoc ? 16 : 13)}
                                        textAnchor="middle"
                                        fontSize={isDoc ? 13 : 10.5}
                                        fontWeight={isDoc ? 600 : 500}
                                        className="fill-slate-700 dark:fill-slate-300"
                                        style={{ pointerEvents: 'none' }}
                                    >
                                        {node.label.length > 24 ? node.label.slice(0, 22) + '…' : node.label}
                                    </text>
                                </g>
                            );
                        })}
                    </g>
                </svg>

                {/* Legend */}
                <div className="absolute bottom-3 left-3 flex flex-wrap items-center gap-3 px-3 py-2 rounded-lg bg-white/85 dark:bg-slate-900/85 backdrop-blur border border-slate-200 dark:border-slate-800 text-[11px] text-slate-600 dark:text-slate-300">
                    <span className="flex items-center gap-1.5">
                        <FileText className="w-3.5 h-3.5 text-primary" /> Document
                    </span>
                    <span className="flex items-center gap-1.5">
                        <Hash className="w-3.5 h-3.5 text-slate-400" /> Topic
                    </span>
                    <span className="flex items-center gap-1.5">
                        <span className="inline-block w-4 border-t-2 border-indigo-500" /> Similar content
                    </span>
                    <span className="flex items-center gap-1.5">
                        <span className="inline-block w-4 border-t border-dashed border-slate-400" /> Mentions
                    </span>
                </div>
            </div>

            <p className="mt-3 text-xs text-slate-400 dark:text-slate-500">
                Drag nodes to rearrange · drag the background to pan · hover a node to spotlight its connections.
            </p>
        </div>
    );
}
