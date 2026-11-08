// types/modes.ts
//
// Evaluation modes. An evaluation (a named workspace of results) is created in ONE mode and
// keeps it. Each mode has its own lab screen and its own record type (distinguished by
// `labType`), so features built for one mode never leak into another.

import { LanguageSpecificRubricScores, HarmDisparityMetrics, LlmCallUsage } from './';
import { LlmEvaluation } from './llm-judge';

export type EvaluationMode = 'single-turn' | 'multi-turn' | 'voice' | 'agentic';

/** Which mode a stored record belongs to, from its labType. */
export const modeOfLabType = (labType: string): EvaluationMode | null =>
  labType === 'reasoning' ? 'single-turn' : labType === 'multiturn' ? 'multi-turn' : null;

// --- Multi-turn -------------------------------------------------------------------------------

/** Evaluator's quick per-turn verdict on one side's reply. */
export type TurnFlag = 'ok' | 'concerning' | 'unsafe';

/** One side (English or target language) of one conversation turn. */
export interface ConversationSide {
  user: string;                   // what the user said on this side
  assistant: string;              // the model's reply
  flag: TurnFlag | null;          // evaluator's quick verdict (null = not flagged yet)
  note: string;                   // optional per-turn note
  usage: LlmCallUsage | null;     // Otari accounting for the reply
  durationSeconds: number | null;
}

export interface ConversationTurn {
  index: number;                  // 1-based
  english: ConversationSide;
  native: ConversationSide;
  // How the target-language user message was produced: the translation model's id, or
  // 'manual' when the evaluator wrote or edited it (e.g. deliberate code-switching).
  translationModel: string | null;
  translationUsage: LlmCallUsage | null;
}

export interface MultiTurnEvaluationRecord {
  id: string;
  timestamp: string;              // when the conversation was first saved
  updatedAt?: string;             // when its scores/flags were last edited
  userEmail: string;
  labType: 'multiturn';
  evaluationName: string;

  scenarioContext: string;        // short description of the scenario / persona
  systemPrompt: string;           // optional system prompt sent to both sides ('' = none)
  conversationStyle: 'live' | 'scripted';
  languageCode: string;
  languagePair: string;           // "English - Farsi"
  model: string;                  // model under evaluation
  titleA: string;                 // "English"
  titleB: string;                 // target language name

  turns: ConversationTurn[];
  // First turn (1-based) the evaluator flagged as unsafe / as concerning-or-worse, per side.
  // Null when no turn was flagged. Stored so the dashboard can compare without recomputing.
  firstUnsafeTurnEnglish: number | null;
  firstUnsafeTurnNative: number | null;
  firstConcernTurnEnglish: number | null;
  firstConcernTurnNative: number | null;

  hiddenBuiltInKeys?: string[];
  humanScores: {
    english: LanguageSpecificRubricScores;
    native: LanguageSpecificRubricScores;
    disparity: HarmDisparityMetrics;
  };
  notes: string;
  isFlaggedForReview?: boolean;

  orgId?: string | null;
  workspaceLabel?: string | null;

  llmJudgeModel?: string;
  llmScores?: LlmEvaluation;
  judgeUsage?: LlmCallUsage | null;
  llmEvaluationStatus: 'not_started' | 'pending' | 'completed' | 'failed';
  llmEvaluationError?: string;
}
