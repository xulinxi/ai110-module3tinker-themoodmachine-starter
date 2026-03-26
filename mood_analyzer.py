# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import string
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        cleaned = text.strip().lower()
        # Remove punctuation from each word so "great!" matches "great"
        tokens = [word.strip(string.punctuation) for word in cleaned.split()]
        # Drop any empty strings left after stripping
        tokens = [t for t in tokens if t]

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        tokens = self.preprocess(text)
        score = 0
        negators = {"not", "no", "never", "don't", "doesn't", "isn't", "wasn't", "aren't"}

        # Emoji sentiment mapping
        positive_emojis = {"😂", "🔥", "❤️", "😍", "🥰", "😊", "🎉", "💪", "👏", "🧡", "💛", "💚", "😎", "🤩"}
        negative_emojis = {"😭", "😢", "😡", "💀", "😤", "🙃", "😞", "😔", "💔", "🥲", "😩"}

        # Negation window: check up to 3 tokens back
        def is_negated(i: int) -> bool:
            start = max(0, i - 4)
            return any(tokens[j] in negators for j in range(start, i))

        for i, token in enumerate(tokens):
            flip = -1 if is_negated(i) else 1
            if token in self.positive_words:
                score += 1 * flip
            elif token in self.negative_words:
                score -= 1 * flip
            elif token in positive_emojis:
                score += 1
            elif token in negative_emojis:
                score -= 1

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.

        Known limitations (fundamental to rule-based approach):
          - Sarcasm is undetectable (e.g. "I love getting stuck in traffic")
          - Multi-word expressions only partially captured ("falling apart")
          - "mixed" requires both positive AND negative words to be present;
            single-sided ambiguity defaults to positive/negative/neutral
        """
        score = self.score_text(text)
        tokens = self.preprocess(text)
        has_pos = any(t in self.positive_words for t in tokens)
        has_neg = any(t in self.negative_words for t in tokens)

        if has_pos and has_neg:
            return "mixed"
        elif score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )

    # -----------------------------------------------------------------
    # Diagnostic output
    # -----------------------------------------------------------------

    def diagnose(self, text: str) -> str:
        """
        Run a breaker-style diagnosis on a single text.

        Reports:
          - Which words affected the score (and how)
          - Which words were ignored
          - Whether one token dominated the decision
        """
        tokens = self.preprocess(text)
        negators = {"not", "no", "never", "don't", "doesn't", "isn't", "wasn't", "aren't"}

        scored_tokens: List[Tuple[str, int]] = []
        ignored_tokens: List[str] = []

        for i, token in enumerate(tokens):
            flip = -1 if (i > 0 and tokens[i - 1] in negators) else 1
            if token in self.positive_words:
                scored_tokens.append((token, 1 * flip))
            elif token in self.negative_words:
                scored_tokens.append((token, -1 * flip))
            elif token not in negators:
                ignored_tokens.append(token)

        total_score = sum(s for _, s in scored_tokens)
        label = self.predict_label(text)

        lines = [
            f'Text:    "{text}"',
            f"Tokens:  {tokens}",
            "",
            "--- Words that affected the score ---",
        ]

        if scored_tokens:
            for word, contribution in scored_tokens:
                sign = "+" if contribution > 0 else ""
                lines.append(f"  {word:15s} → {sign}{contribution}")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append("--- Words that were ignored ---")
        lines.append(f"  {ignored_tokens if ignored_tokens else '(none)'}")

        lines.append("")
        lines.append("--- Dominance check ---")
        if len(scored_tokens) == 1:
            lines.append(
                f'  ⚠ Single token "{scored_tokens[0][0]}" '
                f"fully determined the outcome"
            )
        elif len(scored_tokens) > 1:
            abs_scores = [(w, abs(s)) for w, s in scored_tokens]
            max_abs = max(s for _, s in abs_scores)
            total_abs = sum(s for _, s in abs_scores)
            dominators = [w for w, s in abs_scores if s == max_abs]
            if max_abs > total_abs / 2 and len(dominators) == 1:
                lines.append(
                    f'  ⚠ Token "{dominators[0]}" contributed most of the score'
                )
            else:
                lines.append("  No single token dominated")
        else:
            lines.append("  No scored tokens — label defaults to neutral")

        lines.append("")
        lines.append(f"Score: {total_score} → {label}")
        lines.append("")

        return "\n".join(lines)
