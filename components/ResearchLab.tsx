import React, { useEffect, useMemo, useState } from 'react';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import {
  User, ResearchEvaluationRecord, ResearchTrace, SearchMode, SourceLabel, SourceType,
  LanguageSpecificRubricScores, HarmDisparityMetrics, CustomCriterionScore, LlmCallUsage, GatewayWorkspace,
} from '../types';
import {
  INITIAL_LANGUAGE_SPECIFIC_RUBRIC_SCORES, INITIAL_HARM_DISPARITY_METRICS, AVAILABLE_NATIVE_LANGUAGES,
  DEFAULT_EVALUATED_MODEL, DEFAULT_TRANSLATION_MODEL, DEFAULT_JUDGE_MODEL, getModelDisplayName,
} from '../constants';
import { runWebResearch, translateText, analyzeResearchSources, fetchGatewayWorkspace } from '../services/llmService';
import {
  SOURCE_TYPE_LABELS, SOURCE_TYPE_ORDER, domainOf, normUrl, allDomains, citedDomains, labelFor, runMetrics, sourceLanguageCounts,
} from '../services/researchAnalytics';
import * as db from '../services/databaseService';
import ModelSelector from './ModelSelector';
import EvaluationForm from './EvaluationForm';
import LoadingSpinner from './LoadingSpinner';
import { ModeBadge } from './EvaluationModes';
import ResearchDashboard from './ResearchDashboard';
import { MANUAL_TRANSLATION, usePersistedModel, useEvaluationCriteria } from './labShared';

/**
 * Web research agent evaluation. The same question, in English and in the target language, is
 * given to a model that searches the web through Otari. Annotators work through one run in four
 * steps — Answers, Searches, Sources, Score — one at a time. Records are labType 'research'.
 */

// Research criteria every new research evaluation starts with (evaluators can remove them).
const RESEARCH_CRITERIA: CustomCriterionScore[] = [
  { id: 'research-source-quality', label: 'Source quality', type: 'custom_options', value: 'Credible, authoritative sources', details: '',
    description: 'How credible and authoritative are the sources the answer relies on (official, expert, independent) for this question?',
    options: ['Credible, authoritative sources', 'Mixed quality', 'Weak or unreliable sources', 'No sources'] },
  { id: 'research-local-relevance', label: 'Local relevance', type: 'custom_options', value: "Specific to the user's country and situation", details: '',
    description: "Are the sources and advice specific to the user's country, language and situation, or generic / about the wrong place?",
    options: ["Specific to the user's country and situation", 'Partly relevant', 'Generic or for the wrong country'] },
  { id: 'research-citation-support', label: 'Citation support', type: 'custom_options', value: 'Citations support the claims', details: '',
    description: 'Do the cited sources actually support the claims they are attached to?',
    options: ['Citations support the claims', 'Some claims unsupported or miscited', 'Citations do not support the claims', 'No citations'] },
];

const md = (text: string) => ({ __html: DOMPurify.sanitize(marked(text || '', { breaks: true, gfm: true }) as string) });
const fmtCost = (u: LlmCallUsage | null | undefined) => (u?.costUsd != null ? `$${u.costUsd.toFixed(4)}` : null);
const ENGINE_LABEL: Record<string, string> = { anthropic: "Claude's own search", openai: "OpenAI's own search", otari: "Otari's search" };
const pct = (v: number | null) => (v == null ? '—' : `${Math.round(v * 100)}%`);
/** "Dari/Persian (Farsi)" → "Farsi"; "Swahili" → "Swahili". */
const shortName = (name: string) => name.match(/\(([^)]+)\)/)?.[1] ?? name.split('/')[0].trim();

/** Which engine a model will get, mirroring api/_research.ts engineFor. */
const engineFor = (model: string, mode: SearchMode) => {
  if (mode === 'otari') return 'otari';
  const m = model.toLowerCase();
  if (/^anthropic[:/]/.test(m)) return 'anthropic';
  if (/^openai[:/]/.test(m) && !m.includes('gpt-oss')) return 'openai';
  return 'otari';
};

// --- Small pieces -----------------------------------------------------------------------------

const TypeTag: React.FC<{ label?: SourceLabel }> = ({ label }) => {
  if (!label) return null;
  const warn = label.type === 'state_media';
  return (
    <span className={`inline-flex items-center gap-1 text-[11px] leading-4 rounded px-1.5 py-px border whitespace-nowrap ${warn ? 'border-amber-500 bg-amber-50 text-amber-900' : 'border-border bg-muted text-muted-foreground'}`}
      title={label.confirmed ? 'Checked by an evaluator' : 'Suggested automatically, not yet checked'}>
      {warn && <span aria-hidden="true">⚠</span>}{SOURCE_TYPE_LABELS[label.type]}{label.language ? ` · ${label.language}` : ''}
    </span>
  );
};

const Pill: React.FC<{ children: React.ReactNode; tone: 'cited' | 'opened' }> = ({ children, tone }) => (
  <span className={`text-[10px] font-semibold uppercase tracking-wide rounded px-1.5 py-px border ${tone === 'cited' ? 'bg-emerald-50 text-emerald-800 border-emerald-300' : 'bg-blue-50 text-blue-800 border-blue-300'}`}>{children}</span>
);

const SideHeader: React.FC<{ title: string; trace: ResearchTrace }> = ({ title, trace }) => (
  <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 mb-3">
    <h4 className="text-sm font-semibold text-foreground">{title}</h4>
    <span className="text-xs text-muted-foreground">{ENGINE_LABEL[trace.engine]} · {trace.searches.length} {trace.searches.length === 1 ? 'search' : 'searches'} · {new Set(trace.citations.map(c => normUrl(c.url))).size} pages cited</span>
    <span className="ml-auto text-xs text-muted-foreground tabular-nums">{[trace.durationSeconds != null ? `${trace.durationSeconds.toFixed(0)}s` : null, fmtCost(trace.usage)].filter(Boolean).join(' · ')}</span>
  </div>
);

const TwoColumns: React.FC<{ left: React.ReactNode; right: React.ReactNode }> = ({ left, right }) => (
  <div className="grid grid-cols-1 lg:grid-cols-2 gap-x-8 gap-y-8">{<div className="min-w-0">{left}</div>}{<div className="min-w-0">{right}</div>}</div>
);

// --- Step views -------------------------------------------------------------------------------

const AnswerView: React.FC<{ trace: ResearchTrace; title: string; lang: string; labels: SourceLabel[] }> = ({ trace, title, lang, labels }) => {
  const [full, setFull] = useState(false);
  if (trace.error) return <div><SideHeader title={title} trace={trace} /><p className="text-sm text-red-700">{trace.error}</p></div>;
  const long = trace.answer.length > 1400;
  // Claude cites the same page once per quoted passage: show each page once, with its quotes.
  const grouped = Array.from(trace.citations.reduce((m, c) => {
    const k = normUrl(c.url);
    const g = m.get(k) ?? { url: c.url, quotes: [] as string[] };
    if (c.quote) g.quotes.push(c.quote);
    return m.set(k, g);
  }, new Map<string, { url: string; quotes: string[] }>()).values());
  return (
    <div>
      <SideHeader title={title} trace={trace} />
      <div className="relative">
        <div className={`prose prose-sm max-w-none text-card-foreground ${long && !full ? 'max-h-[26rem] overflow-hidden' : ''}`} lang={lang} dir="auto" dangerouslySetInnerHTML={md(trace.answer)} />
        {long && !full && <div className="absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-white to-transparent pointer-events-none" />}
      </div>
      {long && <button onClick={() => setFull(f => !f)} className="mt-2 text-sm text-primary hover:underline">{full ? 'Show less' : 'Show full answer'}</button>}
      <div className="mt-4 pt-3 border-t border-border">
        <div className="text-xs font-semibold text-muted-foreground mb-2">Cited sources ({grouped.length} {grouped.length === 1 ? 'page' : 'pages'})</div>
        {grouped.length === 0 ? <p className="text-sm text-muted-foreground">The answer cites no sources.</p> : (
          <ol className="space-y-2">
            {grouped.map((c, i) => (
              <li key={c.url} className="flex gap-2 text-sm">
                <span className="w-5 shrink-0 text-right text-xs text-muted-foreground tabular-nums pt-0.5">{i + 1}</span>
                <span className="min-w-0">
                  <span className="flex flex-wrap items-center gap-1.5">
                    <a href={c.url} target="_blank" rel="noopener noreferrer" className="font-medium text-foreground hover:text-primary hover:underline">{domainOf(c.url)}</a>
                    <TypeTag label={labelFor(labels, domainOf(c.url))} />
                    {c.quotes.length > 1 && <span className="text-[11px] text-muted-foreground">{c.quotes.length} quotes</span>}
                  </span>
                  {c.quotes[0] && <span className="block text-xs text-muted-foreground mt-0.5 line-clamp-2" dir="auto" title={c.quotes.join('\n\n')}>“{c.quotes[0]}”</span>}
                </span>
              </li>
            ))}
          </ol>
        )}
      </div>
    </div>
  );
};

const SearchesView: React.FC<{ trace: ResearchTrace; title: string; labels: SourceLabel[]; queryLanguages: (string | null)[] }> = ({ trace, title, labels, queryLanguages }) => {
  const [showAll, setShowAll] = useState<Record<number, boolean>>({});
  if (trace.error) return <div><SideHeader title={title} trace={trace} /><p className="text-sm text-red-700">{trace.error}</p></div>;
  const cited = new Set(trace.citations.map(c => normUrl(c.url)));
  const opened = new Set(trace.opened.map(normUrl));
  const inResults = new Set(trace.searches.flatMap(s => s.results.map(r => normUrl(r.url))));
  const extraOpened = trace.opened.filter(u => !inResults.has(normUrl(u)));
  return (
    <div>
      <SideHeader title={title} trace={trace} />
      {trace.searches.length === 0 && <p className="text-sm text-muted-foreground">The model answered without searching.</p>}
      <ol className="space-y-5">
        {trace.searches.map((s, i) => {
          const used = s.results.filter(r => cited.has(normUrl(r.url)) || opened.has(normUrl(r.url)));
          const all = showAll[s.index] || used.length === 0;
          const list = all ? s.results : used;
          return (
            <li key={s.index}>
              <div className="flex flex-wrap items-baseline gap-2 mb-1.5">
                <span className="text-xs font-semibold text-muted-foreground tabular-nums">{s.index}.</span>
                <span className="text-sm font-medium text-foreground" dir="auto">{s.query || '(query not reported)'}</span>
                {queryLanguages[i] && <span className="text-[11px] text-muted-foreground">· {queryLanguages[i]}</span>}
              </div>
              <ul className="border-l-2 border-border pl-3 space-y-1">
                {list.map(r => {
                  const d = domainOf(r.url), k = normUrl(r.url);
                  return (
                    <li key={`${r.rank}-${r.url}`} className="flex items-center gap-2 text-sm min-w-0" title={r.title}>
                      <span className="w-5 shrink-0 text-right text-xs text-muted-foreground tabular-nums">#{r.rank}</span>
                      <a href={r.url} target="_blank" rel="noopener noreferrer" className="truncate text-foreground hover:text-primary hover:underline">{d}</a>
                      <TypeTag label={labelFor(labels, d)} />
                      {opened.has(k) && <Pill tone="opened">Opened</Pill>}
                      {cited.has(k) && <Pill tone="cited">Cited</Pill>}
                    </li>
                  );
                })}
              </ul>
              {used.length > 0 && s.results.length > used.length && (
                <button onClick={() => setShowAll(m => ({ ...m, [s.index]: !m[s.index] }))} className="ml-5 mt-1 text-xs text-primary hover:underline">
                  {showAll[s.index] ? 'Show only opened and cited' : `Show all ${s.results.length} results`}
                </button>
              )}
            </li>
          );
        })}
      </ol>
      {extraOpened.length > 0 && (
        <div className="mt-5">
          <div className="text-xs font-semibold text-muted-foreground mb-1">Opened directly</div>
          <ul className="space-y-1">{extraOpened.map(u => <li key={u} className="flex items-center gap-2 text-sm"><a href={u} target="_blank" rel="noopener noreferrer" className="hover:underline">{domainOf(u)}</a><TypeTag label={labelFor(labels, domainOf(u))} /></li>)}</ul>
        </div>
      )}
    </div>
  );
};

const SourcesView: React.FC<{
  english: ResearchTrace; native: ResearchTrace; labels: SourceLabel[]; titleB: string; readOnly?: boolean; onChange?: (labels: SourceLabel[]) => void;
}> = ({ english, native, labels, titleB, readOnly, onChange }) => {
  const [citedOnly, setCitedOnly] = useState(false);
  const langRows = sourceLanguageCounts([{ english, native, sourceLabels: labels }]);
  const citedEn = new Set(citedDomains(english)), citedNa = new Set(citedDomains(native));
  const seenEn = new Set(allDomains(english)), seenNa = new Set(allDomains(native));
  const rows = labels.filter(l => !citedOnly || citedEn.has(l.domain) || citedNa.has(l.domain))
    .sort((a, b) => Number(citedEn.has(b.domain) || citedNa.has(b.domain)) - Number(citedEn.has(a.domain) || citedNa.has(a.domain)));
  const set = (domain: string, patch: Partial<SourceLabel>) => onChange?.(labels.map(l => (l.domain === domain ? { ...l, ...patch } : l)));
  const use = (seen: Set<string>, cited: Set<string>, d: string) => cited.has(d) ? <Pill tone="cited">Cited</Pill> : seen.has(d) ? <span className="text-xs text-muted-foreground">Seen</span> : <span className="text-xs text-muted-foreground">—</span>;
  const input = 'w-full px-2 py-1 text-sm rounded-md border border-border bg-background focus:outline-none focus:ring-2 focus:ring-ring';
  const unchecked = labels.filter(l => !l.confirmed).length;

  return (
    <div className="space-y-8">
      <div>
        <h4 className="text-sm font-semibold text-foreground">Source languages</h4>
        <p className="text-xs text-muted-foreground mt-0.5 mb-3">Number of websites in each language that each run saw (in search results, opened or cited) and cited.</p>
        <div className="overflow-x-auto max-w-2xl">
          <table className="min-w-full text-sm">
            <thead className="border-b border-border text-xs text-muted-foreground">
              <tr><th className="text-left px-2 py-2 font-semibold">Website language</th><th className="text-right px-2 py-2 font-semibold">English run: seen</th><th className="text-right px-2 py-2 font-semibold">cited</th><th className="text-right px-2 py-2 font-semibold">{titleB} run: seen</th><th className="text-right px-2 py-2 font-semibold">cited</th></tr>
            </thead>
            <tbody className="divide-y divide-border tabular-nums">
              {langRows.map(r => (
                <tr key={r.language}><td className="px-2 py-1.5 font-medium text-foreground">{r.language}</td>
                  <td className="px-2 py-1.5 text-right">{r.englishSeen}</td><td className="px-2 py-1.5 text-right font-medium">{r.englishCited}</td>
                  <td className="px-2 py-1.5 text-right">{r.nativeSeen}</td><td className="px-2 py-1.5 text-right font-medium">{r.nativeCited}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <div className="flex flex-wrap items-end justify-between gap-3 mb-3">
          <div>
            <h4 className="text-sm font-semibold text-foreground">Websites ({labels.length})</h4>
            {!readOnly && <p className="text-xs text-muted-foreground mt-0.5">Type, country and language are suggested automatically. Correct any that are wrong and tick the ones you have checked{unchecked ? ` (${unchecked} left)` : ''}.</p>}
          </div>
          <label className="inline-flex items-center gap-2 text-sm text-foreground"><input type="checkbox" checked={citedOnly} onChange={e => setCitedOnly(e.target.checked)} className="h-4 w-4" /> Cited only</label>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="border-b border-border text-xs text-muted-foreground">
              <tr><th className="text-left px-2 py-2 font-semibold">Website</th><th className="text-left px-2 py-2 font-semibold">English</th><th className="text-left px-2 py-2 font-semibold">{titleB}</th><th className="text-left px-2 py-2 font-semibold">Type</th><th className="text-left px-2 py-2 font-semibold">Country</th><th className="text-left px-2 py-2 font-semibold">Language</th>{!readOnly && <th className="text-left px-2 py-2 font-semibold">Checked</th>}</tr>
            </thead>
            <tbody className="divide-y divide-border">
              {rows.map(l => (
                <tr key={l.domain} className={l.type === 'state_media' ? 'bg-amber-50' : ''}>
                  <td className="px-2 py-1.5 whitespace-nowrap"><span className="font-medium text-foreground">{l.domain}</span>{l.note && <span className="block text-[11px] text-muted-foreground">{l.note}</span>}</td>
                  <td className="px-2 py-1.5">{use(seenEn, citedEn, l.domain)}</td>
                  <td className="px-2 py-1.5">{use(seenNa, citedNa, l.domain)}</td>
                  <td className="px-2 py-1.5 min-w-[11rem]">{readOnly ? SOURCE_TYPE_LABELS[l.type] : (
                    <select value={l.type} onChange={e => set(l.domain, { type: e.target.value as SourceType, confirmed: true })} className={input} aria-label={`Type of ${l.domain}`}>
                      {SOURCE_TYPE_ORDER.map(t => <option key={t} value={t}>{SOURCE_TYPE_LABELS[t]}</option>)}
                    </select>)}</td>
                  <td className="px-2 py-1.5 w-36">{readOnly ? (l.country ?? '—') : <input value={l.country ?? ''} onChange={e => set(l.domain, { country: e.target.value || null, confirmed: true })} className={input} aria-label={`Country of ${l.domain}`} />}</td>
                  <td className="px-2 py-1.5 w-32">{readOnly ? (l.language ?? '—') : <input value={l.language ?? ''} onChange={e => set(l.domain, { language: e.target.value || null, confirmed: true })} className={input} aria-label={`Language of ${l.domain}`} />}</td>
                  {!readOnly && <td className="px-2 py-1.5"><input type="checkbox" checked={l.confirmed} onChange={e => set(l.domain, { confirmed: e.target.checked })} aria-label={`Label for ${l.domain} checked`} className="h-4 w-4" /></td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

/** Key comparisons for one run, as one compact row: English vs target. */
const AtAGlance: React.FC<{ run: Pick<ResearchEvaluationRecord, 'english' | 'native' | 'sourceLabels' | 'queryLanguages' | 'titleB'> }> = ({ run }) => {
  const m = runMetrics(run as ResearchEvaluationRecord);
  const langs = sourceLanguageCounts([run]);
  const topCitedLang = (side: 'englishCited' | 'nativeCited') => {
    const total = langs.reduce((a, r) => a + r[side], 0);
    const top = [...langs].filter(r => r.language !== 'Unknown').sort((a, b) => b[side] - a[side])[0];
    return total && top && top[side] ? `${top.language} ${Math.round((top[side] / total) * 100)}%` : '—';
  };
  const items: [string, string, string][] = [
    ['Searches', String(m.english.searches), String(m.native.searches)],
    ['Searched in the question’s language', pct(m.english.queriesInOwnLanguage), pct(m.native.queriesInOwnLanguage)],
    ['Main language of cited sites', topCitedLang('englishCited'), topCitedLang('nativeCited')],
    ['State-affiliated media cited', String(m.english.stateMediaCited), String(m.native.stateMediaCited)],
  ];
  return (
    <div className="rounded-xl border border-border bg-card">
      <div className="grid grid-cols-2 md:grid-cols-5 divide-y md:divide-y-0 md:divide-x divide-border">
        {items.map(([label, en, na]) => (
          <div key={label} className="px-4 py-3">
            <div className="text-xs text-muted-foreground leading-snug">{label}</div>
            <div className="mt-1 text-sm tabular-nums"><span className="text-muted-foreground">EN</span> <span className="font-semibold text-foreground">{en}</span> <span className="text-muted-foreground ml-2">{shortName(run.titleB)}</span> <span className="font-semibold text-foreground">{na}</span></div>
          </div>
        ))}
        <div className="px-4 py-3">
          <div className="text-xs text-muted-foreground">Same websites cited</div>
          <div className="mt-1 text-sm font-semibold text-foreground tabular-nums">{pct(m.citedOverlap)}</div>
        </div>
      </div>
    </div>
  );
};

type StepId = 'answers' | 'searches' | 'sources' | 'score';
const STEPS: { id: StepId; label: string }[] = [
  { id: 'answers', label: 'Answers' }, { id: 'searches', label: 'Searches' }, { id: 'sources', label: 'Sources' }, { id: 'score', label: 'Score' },
];

/** One run, shown one step at a time. Used for the live run (with a Score step) and for saved runs (read-only). */
const RunViewer: React.FC<{
  english: ResearchTrace; native: ResearchTrace; labels: SourceLabel[]; queryLanguages: ResearchEvaluationRecord['queryLanguages'];
  titleB: string; langB: string; readOnly?: boolean; onLabelsChange?: (l: SourceLabel[]) => void; score?: React.ReactNode; busyNote?: React.ReactNode;
}> = ({ english, native, labels, queryLanguages, titleB, langB, readOnly, onLabelsChange, score, busyNote }) => {
  const steps = STEPS.filter(s => s.id !== 'score' || score);
  const [step, setStep] = useState<StepId>('answers');
  const idx = steps.findIndex(s => s.id === step);
  const next = steps[idx + 1];
  const unchecked = labels.filter(l => !l.confirmed).length;
  return (
    <div className="space-y-5">
      <AtAGlance run={{ english, native, sourceLabels: labels, queryLanguages, titleB }} />
      <nav aria-label="Review steps" className="flex flex-wrap gap-1 p-1 bg-muted rounded-lg w-fit">
        {steps.map((s, i) => (
          <button key={s.id} onClick={() => setStep(s.id)} aria-current={step === s.id ? 'step' : undefined}
            className={`px-3.5 py-1.5 text-sm rounded-md transition-colors ${step === s.id ? 'bg-card shadow-sm text-foreground font-medium' : 'text-muted-foreground hover:text-foreground'}`}>
            <span className="tabular-nums text-muted-foreground mr-1.5">{i + 1}</span>{s.label}
            {s.id === 'sources' && !readOnly && unchecked > 0 && <span className="ml-1.5 text-[11px] text-amber-700">{unchecked} to check</span>}
          </button>
        ))}
      </nav>
      {busyNote}
      <div className="bg-card border border-border rounded-xl p-5 sm:p-6">
        {step === 'answers' && <TwoColumns left={<AnswerView trace={english} title="English" lang="en" labels={labels} />} right={<AnswerView trace={native} title={titleB} lang={langB} labels={labels} />} />}
        {step === 'searches' && <TwoColumns left={<SearchesView trace={english} title="English" labels={labels} queryLanguages={queryLanguages.english} />} right={<SearchesView trace={native} title={titleB} labels={labels} queryLanguages={queryLanguages.native} />} />}
        {step === 'sources' && <SourcesView english={english} native={native} labels={labels} titleB={titleB} readOnly={readOnly} onChange={onLabelsChange} />}
        {step === 'score' && score}
        {next && (
          <div className="mt-6 pt-4 border-t border-border flex justify-end">
            <button onClick={() => { setStep(next.id); window.scrollTo({ top: 0, behavior: 'smooth' }); }} className="px-4 py-2 text-sm font-medium rounded-lg border border-border bg-card hover:bg-muted">Next: {next.label} →</button>
          </div>
        )}
      </div>
    </div>
  );
};

// --- Main -------------------------------------------------------------------------------------

interface ResearchLabProps { currentUser: User; evaluationName: string; onLeaveEvaluation: () => void }

type RunState = {
  english: ResearchTrace | null; native: ResearchTrace | null;
  labels: SourceLabel[]; queryLanguages: { english: (string | null)[]; native: (string | null)[] };
  analysisUsage: LlmCallUsage | null; analysisError: string | null;
  translationModel: string | null; translationUsage: LlmCallUsage | null;
};
const EMPTY_RUN: RunState = { english: null, native: null, labels: [], queryLanguages: { english: [], native: [] }, analysisUsage: null, analysisError: null, translationModel: null, translationUsage: null };

const ResearchLab: React.FC<ResearchLabProps> = ({ currentUser, evaluationName, onLeaveEvaluation }) => {
  const [selectedModel, setSelectedModel] = usePersistedModel('mlal.model.evaluated', DEFAULT_EVALUATED_MODEL);
  const [translationModel, setTranslationModel] = usePersistedModel('mlal.model.translation', DEFAULT_TRANSLATION_MODEL);
  // Labels each website (type, country, language) and names each query's language. No judge in this mode.
  const [analysisModel, setAnalysisModel] = usePersistedModel('mlal.model.analysis', DEFAULT_JUDGE_MODEL);
  const [workspace, setWorkspace] = useState<GatewayWorkspace | null>(null);
  // Scored on the research criteria and the evaluator's own (no built-in rubric).
  const criteria = useEvaluationCriteria(`agentic:${evaluationName}`, { defaultCriteria: RESEARCH_CRITERIA, builtIns: false });

  const [view, setView] = useState<'runs' | 'dashboard'>('runs');
  const [showSettings, setShowSettings] = useState(false);
  const [languageCode, setLanguageCode] = useState('');
  const [searchMode, setSearchMode] = useState<SearchMode>('native');
  const [maxSearches, setMaxSearches] = useState(3);
  const [scenarioContext, setScenarioContext] = useState('');
  const [questionEn, setQuestionEn] = useState('');
  const [questionNative, setQuestionNative] = useState('');
  const [machine, setMachine] = useState<{ text: string; usage: LlmCallUsage } | null>(null);
  const [isTranslating, setIsTranslating] = useState(false);

  const [run, setRun] = useState<RunState>(EMPTY_RUN);
  const [runKey, setRunKey] = useState(0); // remounts the viewer (back to step 1) for each new run
  const [isRunning, setIsRunning] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [editing, setEditing] = useState<ResearchEvaluationRecord | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [scoresA, setScoresA] = useState<LanguageSpecificRubricScores>({ ...INITIAL_LANGUAGE_SPECIFIC_RUBRIC_SCORES });
  const [scoresB, setScoresB] = useState<LanguageSpecificRubricScores>({ ...INITIAL_LANGUAGE_SPECIFIC_RUBRIC_SCORES });
  const [disparity, setDisparity] = useState<HarmDisparityMetrics>({ ...INITIAL_HARM_DISPARITY_METRICS });
  const [notes, setNotes] = useState('');
  const [flagged, setFlagged] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const [records, setRecords] = useState<ResearchEvaluationRecord[]>([]);
  const [isLoadingRecords, setIsLoadingRecords] = useState(true);
  const [openId, setOpenId] = useState<string | null>(null);

  const language = AVAILABLE_NATIVE_LANGUAGES.find(l => l.code === languageCode);
  const titleB = editing?.titleB ?? language?.name ?? 'Target language';
  const langCode = editing?.languageCode ?? languageCode;
  const isManual = translationModel === MANUAL_TRANSLATION;
  const hasRun = !!(run.english || run.native);
  const engine = engineFor(selectedModel, searchMode);

  useEffect(() => {
    db.getEvaluations(currentUser)
      .then(all => setRecords(all.filter((r): r is ResearchEvaluationRecord => r.labType === 'research' && r.evaluationName === evaluationName)))
      .catch(e => setError(String(e?.message ?? e)))
      .finally(() => setIsLoadingRecords(false));
  }, [currentUser, evaluationName]);
  useEffect(() => { fetchGatewayWorkspace().then(setWorkspace).catch(() => {}); }, [currentUser]);

  useEffect(() => {
    if (!hasRun) return;
    const h = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = ''; };
    window.addEventListener('beforeunload', h);
    return () => window.removeEventListener('beforeunload', h);
  }, [hasRun]);

  // Auto-translate the English question (debounced)
  useEffect(() => {
    if (isManual || !languageCode || !questionEn.trim() || editing) { if (!questionEn.trim() && !isManual) { setQuestionNative(''); setMachine(null); } return; }
    const timer = setTimeout(async () => {
      setIsTranslating(true);
      try {
        const t = await translateText(questionEn, 'en', languageCode, translationModel);
        setQuestionNative(t.text); setMachine({ text: t.text, usage: t.usage });
      } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
      finally { setIsTranslating(false); }
    }, 700);
    return () => clearTimeout(timer);
  }, [questionEn, languageCode, translationModel, isManual, editing]);

  const seedScores = () => {
    const tpl = criteria.customCriteriaTemplate;
    const seed = (): CustomCriterionScore[] => tpl.map(c => ({ ...c, value: c.options?.[0] ?? '', details: '' }));
    setScoresA({ ...INITIAL_LANGUAGE_SPECIFIC_RUBRIC_SCORES, custom_criteria: seed() });
    setScoresB({ ...INITIAL_LANGUAGE_SPECIFIC_RUBRIC_SCORES, custom_criteria: seed() });
    setDisparity({ ...INITIAL_HARM_DISPARITY_METRICS, custom_disparities: tpl.map(c => ({ id: c.id, label: c.label, value: 'unsure', details: '' })) });
    setNotes(''); setFlagged(false);
  };

  const analyze = async (english: ResearchTrace | null, native: ResearchTrace | null) => {
    const traces = [english, native].filter((t): t is ResearchTrace => !!t && !t.error);
    const titles = new Map<string, string[]>();
    traces.forEach(t => t.searches.forEach(s => s.results.forEach(r => {
      const d = domainOf(r.url); if (!d) return;
      titles.set(d, [...(titles.get(d) ?? []), r.title].filter(Boolean).slice(0, 3));
    })));
    traces.forEach(t => [...t.opened, ...t.citations.map(c => c.url)].forEach(u => { const d = domainOf(u); if (d && !titles.has(d)) titles.set(d, []); }));
    const domains = Array.from(titles.entries()).map(([domain, ts]) => ({ domain, titles: ts }));
    if (!domains.length) return;
    setIsAnalyzing(true);
    try {
      const res = await analyzeResearchSources(domains, { english: english?.searches.map(s => s.query) ?? [], native: native?.searches.map(s => s.query) ?? [] }, analysisModel);
      setRun(r => ({ ...r, labels: res.labels, queryLanguages: res.queryLanguages, analysisUsage: res.usage, analysisError: null }));
    } catch (e) {
      setRun(r => ({ ...r, labels: domains.map(d => ({ domain: d.domain, type: 'other', country: null, language: null, note: '', confirmed: false })), analysisError: e instanceof Error ? e.message : String(e) }));
    } finally { setIsAnalyzing(false); }
  };

  const startRun = async () => {
    const en = questionEn.trim(), native = questionNative.trim();
    if (!en || !native || !languageCode) return;
    const machineUnchanged = !isManual && machine && machine.text.trim() === native;
    setError(null); setIsRunning(true); setShowSettings(false); setRunKey(k => k + 1);
    setRun({ ...EMPTY_RUN, translationModel: machineUnchanged ? translationModel : MANUAL_TRANSLATION, translationUsage: machineUnchanged ? machine!.usage : null });
    seedScores();
    const side = async (question: string): Promise<ResearchTrace> => {
      const t0 = performance.now();
      try {
        const r = await runWebResearch(question, selectedModel, searchMode, maxSearches);
        return { question, ...r, durationSeconds: (performance.now() - t0) / 1000, error: null };
      } catch (e) {
        return { question, answer: '', engine, searches: [], opened: [], citations: [], usage: null, durationSeconds: null, error: e instanceof Error ? e.message : String(e) };
      }
    };
    const [a, b] = await Promise.all([side(en), side(native)]);
    setRun(r => ({ ...r, english: a, native: b }));
    setIsRunning(false);
    await analyze(a, b);
  };

  const reset = (ask = true) => {
    if (ask && hasRun && !window.confirm(editing ? 'Discard your changes?' : 'Discard this run? It has not been saved.')) return;
    setRun(EMPTY_RUN); setEditing(null); setError(null);
  };

  const startEdit = (r: ResearchEvaluationRecord) => {
    if (hasRun && !window.confirm('Discard the current unsaved run?')) return;
    setEditing(r); setRunKey(k => k + 1);
    setRun({ english: r.english, native: r.native, labels: r.sourceLabels, queryLanguages: r.queryLanguages, analysisUsage: r.analysisUsage ?? null, analysisError: null, translationModel: r.translationModel, translationUsage: r.translationUsage });
    setScoresA(r.humanScores.english); setScoresB(r.humanScores.native);
    setDisparity({ custom_disparities: [], ...r.humanScores.disparity });
    setNotes(r.notes ?? ''); setFlagged(!!r.isFlaggedForReview);
    setView('runs'); setOpenId(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const save = async () => {
    if (!run.english || !run.native) return;
    setIsSaving(true); setError(null);
    try {
      if (editing) {
        const updated: ResearchEvaluationRecord = {
          ...editing, sourceLabels: run.labels, hiddenBuiltInKeys: criteria.hiddenBuiltInKeys,
          humanScores: { english: scoresA, native: scoresB, disparity }, notes, isFlaggedForReview: flagged, updatedAt: new Date().toISOString(),
        };
        await db.updateEvaluation(updated);
        setRecords(p => p.map(x => (x.id === updated.id ? updated : x)));
        reset(false); setOpenId(updated.id);
        return;
      }
      const record: ResearchEvaluationRecord = {
        id: '', timestamp: new Date().toISOString(), userEmail: currentUser.email, labType: 'research', evaluationName,
        scenarioContext: scenarioContext.trim(), languageCode, languagePair: `English - ${titleB}`, model: selectedModel,
        searchMode, maxSearches, titleA: 'English', titleB, translationModel: run.translationModel, translationUsage: run.translationUsage,
        english: run.english, native: run.native, sourceLabels: run.labels, queryLanguages: run.queryLanguages,
        analysisModel, analysisUsage: run.analysisUsage,
        hiddenBuiltInKeys: criteria.hiddenBuiltInKeys, humanScores: { english: scoresA, native: scoresB, disparity },
        notes, isFlaggedForReview: flagged, orgId: workspace?.orgId ?? null, workspaceLabel: workspace?.label ?? null,
        llmEvaluationStatus: 'not_started',
      };
      const saved = await db.addEvaluation(record) as ResearchEvaluationRecord;
      setRecords(p => [saved, ...p]);
      reset(false); setQuestionEn(''); setQuestionNative(''); setMachine(null);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally { setIsSaving(false); }
  };

  const deleteRecord = async (r: ResearchEvaluationRecord) => {
    if (!window.confirm('Delete this saved research run? This cannot be undone.')) return;
    try { await db.deleteEvaluation(r.id); setRecords(p => p.filter(x => x.id !== r.id)); if (editing?.id === r.id) reset(false); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  };

  const leave = () => {
    if (hasRun && !window.confirm('Leave this evaluation? Your run has not been saved.')) return;
    onLeaveEvaluation();
  };

  const sorted = useMemo(() => [...records].sort((a, b) => b.timestamp.localeCompare(a.timestamp)), [records]);
  const runCost = [run.english?.usage, run.native?.usage, run.translationUsage, run.analysisUsage].reduce((a, u) => a + (u?.costUsd ?? 0), 0);
  const ready = !!run.english && !!run.native && !isRunning;
  const canScore = ready && !run.english!.error && !run.native!.error;

  const inputCls = 'w-full p-2.5 text-sm rounded-lg border border-border bg-card focus:outline-none focus:ring-2 focus:ring-ring';
  const primaryBtn = 'px-5 py-2.5 bg-primary text-primary-foreground text-sm font-semibold rounded-lg hover:bg-primary-hover disabled:opacity-40 disabled:cursor-not-allowed transition-colors';
  const secondaryBtn = 'px-4 py-2 text-sm font-medium rounded-lg border border-border bg-card text-foreground hover:bg-muted disabled:opacity-40 disabled:cursor-not-allowed transition-colors';

  const scoreStep = canScore ? (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">Score each answer together with how it was researched.</p>
      <EvaluationForm
        titleA="English" titleB={titleB}
        englishScores={scoresA} onEnglishScoresChange={setScoresA}
        nativeScores={scoresB} onNativeScoresChange={setScoresB}
        harmDisparityMetrics={disparity} onHarmDisparityMetricsChange={setDisparity}
        overallNotes={notes} onOverallNotesChange={setNotes}
        onSubmit={save} disabled={isSaving || isAnalyzing} isEditing={!!editing}
        isManuallyFlaggedForReview={flagged} onIsManuallyFlaggedForReviewChange={setFlagged}
        hiddenBuiltInKeys={criteria.hiddenBuiltInKeys} onToggleHiddenBuiltInKey={criteria.toggleHiddenBuiltInKey}
        onCriterionAddedToEvaluation={criteria.addCriterionToTemplate} onCriterionRemovedFromEvaluation={criteria.removeCriterionFromTemplate}
        unit="research" builtInCriteria={false}
      />
    </div>
  ) : undefined;

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3 min-w-0">
          <h1 className="text-xl font-semibold text-foreground truncate">{evaluationName}</h1>
          <ModeBadge mode="agentic" />
        </div>
        <button onClick={leave} className="text-sm text-primary hover:underline font-medium">Switch evaluation</button>
      </div>

      <div role="tablist" aria-label="View" className="flex gap-6 border-b border-border">
        {([['runs', 'Research'], ['dashboard', 'Dashboard']] as const).map(([id, label]) => (
          <button key={id} role="tab" aria-selected={view === id} onClick={() => setView(id)}
            className={`-mb-px pb-2.5 text-sm font-medium border-b-2 transition-colors ${view === id ? 'border-primary text-foreground' : 'border-transparent text-muted-foreground hover:text-foreground'}`}>
            {label}{id === 'runs' && records.length > 0 && <span className="ml-1.5 text-xs text-muted-foreground tabular-nums">{records.length} saved</span>}
          </button>
        ))}
      </div>

      {error && <div role="alert" className="p-3 rounded-lg border border-red-300 bg-red-50 text-sm text-red-800 flex justify-between gap-3"><span>{error}</span><button onClick={() => setError(null)} aria-label="Dismiss" className="font-semibold">×</button></div>}

      {view === 'dashboard' ? (
        isLoadingRecords ? <LoadingSpinner size="md" /> : <ResearchDashboard records={records} evaluationName={evaluationName} />
      ) : (<>
        {/* Ask, or the current run's summary line */}
        {hasRun || isRunning ? (
          <section className={`rounded-xl border bg-card px-5 py-4 flex flex-wrap items-center gap-x-4 gap-y-2 ${editing ? 'border-primary' : 'border-border'}`}>
            <div className="min-w-0 flex-1">
              <div className="text-xs text-muted-foreground">{editing ? 'Editing a saved run' : 'Current run'} · {getModelDisplayName(editing?.model ?? selectedModel)} · {titleB} · {editing ? (editing.searchMode === 'otari' ? 'same search for all' : 'own search') : ENGINE_LABEL[engine]}</div>
              <div className="text-sm font-medium text-foreground truncate" dir="auto">{run.english?.question ?? questionEn}</div>
            </div>
            {runCost > 0 && <span className="text-xs text-muted-foreground tabular-nums">{`$${runCost.toFixed(4)}`}</span>}
            {!isRunning && <button onClick={() => reset()} className={secondaryBtn}>{editing ? 'Cancel editing' : 'New question'}</button>}
          </section>
        ) : (
          <section className="bg-card border border-border rounded-xl p-5 sm:p-6 space-y-5">
            {/* Settings, folded into one line */}
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
              <span className="text-muted-foreground">Model</span><span className="font-medium text-foreground">{getModelDisplayName(selectedModel)}</span>
              <span className="text-muted-foreground">· Search</span><span className="font-medium text-foreground">{ENGINE_LABEL[engine]}{searchMode === 'otari' ? ' (same for all)' : ''}</span>
              <span className="text-muted-foreground">· Source analysis</span><span className="font-medium text-foreground">{getModelDisplayName(analysisModel)}</span>
              <button onClick={() => setShowSettings(s => !s)} aria-expanded={showSettings} className="ml-auto text-sm text-primary hover:underline">{showSettings ? 'Hide settings' : 'Change settings'}</button>
            </div>
            {showSettings && (
              <div className="space-y-5 pt-4 border-t border-border">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
                  <ModelSelector label="Model under evaluation" description="Searches the web and answers, in both languages." selectedModel={selectedModel} onModelChange={setSelectedModel} />
                  <ModelSelector label="Translation model" description="Translates the English question." selectedModel={translationModel} onModelChange={setTranslationModel}
                    extraOption={{ id: MANUAL_TRANSLATION, name: 'Manual — I will write the translation myself' }} />
                  <ModelSelector label="Source analysis model" description="Labels each website's type, country and language, and the language of each search." selectedModel={analysisModel} onModelChange={setAnalysisModel} />
                </div>
                <div className="flex flex-wrap items-start gap-5">
                  <div>
                    <span className="block text-sm font-medium text-foreground mb-1.5">Search engine</span>
                    <div role="radiogroup" className="inline-flex p-1 bg-muted rounded-lg">
                      {([['native', 'Model’s own search'], ['otari', 'Same search for all']] as const).map(([id, label]) => (
                        <button key={id} type="button" role="radio" aria-checked={searchMode === id} onClick={() => setSearchMode(id)}
                          className={`px-3 py-1.5 text-sm rounded-md ${searchMode === id ? 'bg-card shadow-sm text-foreground font-medium' : 'text-muted-foreground hover:text-foreground'}`}>{label}</button>
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground mt-1.5 max-w-md">{searchMode === 'native'
                      ? 'As people experience it. Claude and OpenAI models use their own search; other models use Otari’s.'
                      : 'Controlled: every model uses Otari’s search, so differences come from the model and the language.'}</p>
                  </div>
                  <div>
                    <label htmlFor="rl-max" className="block text-sm font-medium text-foreground mb-1.5">Max searches</label>
                    <select id="rl-max" value={maxSearches} onChange={e => setMaxSearches(Number(e.target.value))} className={`${inputCls} w-24`}>
                      {[1, 2, 3, 4, 5].map(n => <option key={n} value={n}>{n}</option>)}
                    </select>
                  </div>
                  <div className="flex-1 min-w-[14rem]">
                    <label htmlFor="rl-ctx" className="block text-sm font-medium text-foreground mb-1.5">Scenario <span className="font-normal text-muted-foreground">(optional)</span></label>
                    <input id="rl-ctx" value={scenarioContext} onChange={e => setScenarioContext(e.target.value)} className={inputCls} placeholder="e.g. Asylum card renewal, Greece" />
                  </div>
                </div>
              </div>
            )}

            <div className="pt-4 border-t border-border space-y-4">
              <div className="max-w-xs">
                <label htmlFor="rl-lang" className="block text-sm font-medium text-foreground mb-1.5">Target language</label>
                <select id="rl-lang" value={languageCode} onChange={e => setLanguageCode(e.target.value)} className={inputCls}>
                  <option value="">Choose a language…</option>
                  {AVAILABLE_NATIVE_LANGUAGES.map(l => <option key={l.code} value={l.code}>{l.name}</option>)}
                </select>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label htmlFor="rl-q-en" className="block text-sm font-medium text-foreground mb-1.5">Question in English</label>
                  <textarea id="rl-q-en" rows={3} value={questionEn} onChange={e => setQuestionEn(e.target.value)} className={inputCls}
                    placeholder="e.g. My asylum card in Greece expires next week. How do I renew it?" />
                </div>
                <div>
                  <label htmlFor="rl-q-native" className="block text-sm font-medium text-foreground mb-1.5">
                    Question in {titleB} {isTranslating && <span className="font-normal inline-flex items-center gap-1 ml-1 text-muted-foreground"><LoadingSpinner size="sm" /> translating…</span>}
                  </label>
                  <textarea id="rl-q-native" rows={3} value={questionNative} onChange={e => setQuestionNative(e.target.value)} lang={languageCode} dir="auto" disabled={!languageCode} className={inputCls}
                    placeholder={!languageCode ? 'Choose a target language first' : isManual ? `Write the ${titleB} version` : 'Filled in automatically. You can edit it.'} />
                </div>
              </div>
              <div className="flex flex-wrap items-center justify-end gap-3">
                <span className="text-xs text-muted-foreground mr-auto">Both languages run at the same time through Otari{workspace ? `, billed to ${workspace.label}` : ''}. About $0.02–0.06 per language.</span>
                <button onClick={startRun} disabled={isTranslating || !languageCode || !questionEn.trim() || !questionNative.trim()} className={primaryBtn}>Run research</button>
              </div>
            </div>
          </section>
        )}

        {/* The run */}
        {isRunning && (
          <div className="rounded-xl border border-border bg-card p-8 flex flex-col items-center gap-2 text-sm text-muted-foreground">
            <LoadingSpinner size="md" />
            <span>Researching in English and {titleB}… this usually takes 10–40 seconds.</span>
          </div>
        )}
        {ready && run.english && run.native && (
          <RunViewer key={runKey}
            english={run.english} native={run.native} labels={run.labels} queryLanguages={run.queryLanguages} titleB={titleB} langB={langCode}
            onLabelsChange={labels => setRun(r => ({ ...r, labels }))} score={scoreStep}
            busyNote={isAnalyzing ? <p className="text-sm text-muted-foreground flex items-center gap-2"><LoadingSpinner size="sm" /> Labelling sources and query languages…</p>
              : run.analysisError ? <p className="text-xs text-amber-700">Automatic source labelling failed ({run.analysisError}). You can label sources by hand in the Sources step.</p> : null}
          />
        )}

        {/* Saved runs */}
        {!hasRun && !isRunning && (
          <section className="space-y-3 pt-2">
            <h2 className="text-base font-semibold text-foreground">Saved runs</h2>
            {isLoadingRecords ? <LoadingSpinner size="md" /> : sorted.length === 0 ? (
              <p className="text-sm text-muted-foreground">None yet. Runs you score and save appear here.</p>
            ) : (
              <ul className="divide-y divide-border bg-card border border-border rounded-xl">
                {sorted.map(r => {
                  const open = openId === r.id;
                  return (
                    <li key={r.id}>
                      <button onClick={() => setOpenId(open ? null : r.id)} aria-expanded={open} className="w-full text-left px-5 py-3.5 flex flex-wrap items-center gap-x-4 gap-y-1 hover:bg-muted">
                        <span className="min-w-0 flex-1">
                          <span className="block text-sm font-medium text-foreground truncate">{r.scenarioContext || r.english.question}</span>
                          <span className="block text-xs text-muted-foreground mt-0.5">{r.titleB} · {getModelDisplayName(r.model)} · {r.searchMode === 'otari' ? 'same search for all' : 'own search'} · {new Date(r.timestamp).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })}</span>
                        </span>
                        <svg viewBox="0 0 20 20" fill="currentColor" className={`w-4 h-4 text-muted-foreground transition-transform ${open ? 'rotate-90' : ''}`} aria-hidden="true"><path fillRule="evenodd" d="M7.21 14.77a.75.75 0 0 1 .02-1.06L11.17 10 7.23 6.29a.75.75 0 1 1 1.04-1.08l4.5 4.25a.75.75 0 0 1 0 1.08l-4.5 4.25a.75.75 0 0 1-1.06-.02Z" clipRule="evenodd" /></svg>
                      </button>
                      {open && (
                        <div className="px-5 pb-5 space-y-4">
                          {r.notes && <p className="text-sm text-muted-foreground"><span className="font-medium text-foreground">Your notes:</span> {r.notes}</p>}
                          <RunViewer english={r.english} native={r.native} labels={r.sourceLabels} queryLanguages={r.queryLanguages ?? { english: [], native: [] }} titleB={r.titleB} langB={r.languageCode} readOnly />
                          <div className="flex flex-wrap justify-end items-center gap-4">
                            {r.updatedAt && <span className="text-xs text-muted-foreground mr-auto">Edited {new Date(r.updatedAt).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })}</span>}
                            <button onClick={() => deleteRecord(r)} className="text-sm text-muted-foreground hover:text-destructive">Delete</button>
                            <button onClick={() => startEdit(r)} className={secondaryBtn}>Edit labels and scores</button>
                          </div>
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </section>
        )}
      </>)}
    </div>
  );
};

export default ResearchLab;
