# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

## 1. Model Overview

**Model type:**
I compared both models: a rule based mood analyzer and an ML classifier using logistic regression.

**Intended purpose:**
Classify short text messages (social media posts, casual messages) into mood labels: positive, negative, neutral, or mixed.

**How it works (brief):**
- Rule based: Preprocesses text into tokens, matches each token against positive/negative word lists, applies negation detection (4-token window), scores emojis, and sums up a numeric score. If both positive and negative words are present, it returns "mixed." Otherwise, score > 0 is positive, score < 0 is negative, and score == 0 is neutral.
- ML model: Converts text into numeric vectors using CountVectorizer (bag of words), then trains a LogisticRegression classifier on the labeled dataset. It learns word-to-label associations from the data rather than relying on hand-built rules.

## 2. Data

**Dataset description:**
The dataset contains 27 labeled posts in `SAMPLE_POSTS`. It started with 6 starter posts and was expanded with 21 additional examples covering slang, sarcasm, emojis, negation, and mixed emotions.

**Labeling process:**
Labels were assigned by reading each post and deciding what mood a human reader would most likely interpret. Posts that were hard to label:
- "I love getting stuck in traffic" -- sarcasm makes this negative despite positive words
- "I absolutely love waking up at 5am for no reason" -- sarcastic, labeled negative
- "everything's falling apart but at least I have coffee" -- could be negative or mixed; labeled mixed because the speaker acknowledges a silver lining
- "just submitted my project idk if it's good but it's DONE" -- relief mixed with uncertainty

**Important characteristics of your dataset:**
- Contains slang ("ngl", "no cap", "lowkey", "sick", "wicked")
- Contains emojis (😭, 🔥, 🙃, 😊, 💀, 🧡)
- Includes sarcasm ("I love getting stuck in traffic", "I absolutely love waking up at 5am")
- Some posts express mixed feelings ("exhausted but proud", "hate how much I love")
- Contains short or ambiguous messages ("meh", "This is fine", "it's whatever I guess")

**Possible issues with the dataset:**
- Very small (27 examples) -- not enough for the ML model to generalize to unseen text
- Label imbalance: more negative and positive examples than neutral or mixed
- Written by one person, so language style and slang are not diverse
- No posts from different cultural or regional language communities
- Sarcasm labels rely on subjective interpretation

## 3. How the Rule Based Model Works

**Your scoring rules:**
- Each positive word adds +1 to the score; each negative word subtracts 1
- Negation handling: if a negator ("not", "no", "never", "don't", "doesn't", "isn't", "wasn't", "aren't") appears within 4 tokens before a sentiment word, the word's score is flipped
- Emoji scoring: positive emojis (😂, 🔥, ❤️, 😍, 😊, etc.) add +1; negative emojis (😭, 😢, 😡, 🙃, 💀, etc.) subtract 1
- Punctuation is stripped during preprocessing so "great!" matches "great"
- Label mapping: if both positive and negative words are present, return "mixed"; otherwise score > 0 is "positive", score < 0 is "negative", score == 0 is "neutral"

**Strengths of this approach:**
- Fully transparent and explainable -- the `diagnose` method shows exactly which words scored and which were ignored
- Predictable behavior -- same input always produces the same output
- Works with zero training data
- Handles simple negation well ("not happy" -> negative, "not bad" -> positive)
- Easy to extend by adding words to the lists

**Weaknesses of this approach:**
- Cannot detect sarcasm ("I love getting stuck in traffic" reads as mixed, not negative)
- Limited to words explicitly in the word lists -- any unlisted word is invisible
- "Mixed" label requires both a positive AND negative word to be present; single-sided ambiguity defaults to a simpler label
- Multi-word expressions ("falling apart") only partially captured -- each word scored independently
- The wider negation window (4 tokens) can cause false flips in longer sentences

## 4. How the ML Model Works

**Features used:**
Bag of words using CountVectorizer. Each text is represented as a vector of word counts.

**Training data:**
The model trained on all 27 posts in `SAMPLE_POSTS` with labels from `TRUE_LABELS`.

**Training behavior:**
When the dataset grew from 13 to 21 to 27 examples, the ML model's training accuracy stayed at 100%. Adding more diverse examples (sarcasm, slang, mixed emotions) gave it more patterns to learn from. The model picked up associations that rule-based systems miss, like the word "traffic" appearing in negative-labeled posts.

**Strengths and weaknesses:**
Strengths:
- Learns patterns automatically from the data -- no need to manually build word lists
- Can pick up subtle word associations (e.g., "traffic" correlates with negative mood)
- Handles sarcasm if similar examples appear in training data

Weaknesses:
- 100% training accuracy is misleading -- the model is evaluated on data it has already memorized
- With only 27 examples, it will likely fail on new sentences with unfamiliar vocabulary
- May learn spurious correlations (e.g., "5am" = negative just because it appeared in one sarcastic post)
- Less transparent -- harder to explain why a specific prediction was made

### Comparison: Rule Based vs ML

**Did the ML model behave differently?**
Yes, significantly. The ML model scored 27/27 (100%) on the training data while the rule based model scored 20/27 (74%). The ML model correctly classified every post it was trained on, including the 7 posts the rule based model got wrong. However, this comparison is not entirely fair -- the ML model was tested on the same data it learned from, so it had an inherent advantage.

**Failures the ML model fixed:**
- "I love getting stuck in traffic" -- the rule based model returned "mixed" because it saw "love" (+1) and "stuck" (-1). The ML model learned that this specific combination of words maps to "negative" from the training label.
- "best day ever honestly" -- the rule based model returned "neutral" because none of those words are in the word lists. The ML model learned from the bag of words that "best" and "ever" together signal positive.
- "I'm stressed but at least it's almost Friday" -- the rule based model returned "negative" because only "stressed" matched. The ML model learned the full word pattern and returned "mixed."

**Did the ML model introduce new failures?**
Not on the training data, but it would on new inputs. For example, a sentence like "what a lovely disaster" would challenge the ML model if it never saw "disaster" paired with a positive word during training. The rule based model would at least catch "lovely" (if added to the list) and "disaster" (if added), and return "mixed." The ML model has no fallback logic -- it either learned the pattern or it guesses.

**How sensitive was it to the labels?**
Very sensitive. The ML model treats the labels as ground truth and optimizes entirely around them. This means:
- If a label is wrong (e.g., labeling a sarcastic post as "positive" by mistake), the ML model will learn that mistake as fact.
- Changing a single label (e.g., flipping "everything's falling apart but at least I have coffee" from "mixed" to "negative") could shift how the model weights words like "coffee" or "least" across all future predictions.
- The rule based model is not sensitive to labels at all -- it uses hand-built word lists and ignores `TRUE_LABELS` entirely. Its behavior only changes when you edit the word lists or scoring logic.

This label sensitivity is the core trade-off: the ML model is more powerful because it learns from data, but it is also more fragile because it trusts whatever data you give it.

## 5. Evaluation

**How you evaluated the model:**
Both models were evaluated on the 27 labeled posts in `dataset.py`. The rule based model scored each post and compared the predicted label to the true label. The ML model was evaluated on its own training data.

- Rule based accuracy: 20/27 (74%)
- ML model accuracy: 27/27 (100%, but on training data)

**Examples of correct predictions:**
- "I love this class so much" -> positive (both models). The word "love" is a clear positive signal.
- "Not bad actually" -> positive (both models). The rule based model correctly flipped "bad" using the negation handler.
- "I'm exhausted but proud of myself" -> mixed (both models). Both "exhausted" (negative) and "proud" (positive) are present, triggering the mixed label in the rule based model.

**Examples of incorrect predictions (rule based):**
- "I love getting stuck in traffic" -> predicted mixed, true label negative. The model sees "love" (positive) and "stuck" (negative) and returns "mixed," but a human recognizes this as sarcasm.
- "best day ever honestly" -> predicted neutral, true label positive. None of the words "best", "day", "ever", or "honestly" are in the positive word list, so the model scores it as 0.
- "this is so frustrating I want to scream" -> predicted neutral, true label negative. "Frustrating" and "scream" are not in the negative word list.

The ML model got all of these correct because it learned from the labeled training data.

## 6. Limitations

### Sarcasm is undetectable by the rule based model

"I love getting stuck in traffic" -> predicted **mixed**, true label **negative**.
The model found "love" (+1) and "stuck" (-1), so both word lists matched and it returned "mixed." A human immediately reads this as sarcastic -- nobody loves traffic. But the rule based model has no concept of intent or context; it scores words in isolation. The same failure occurs with "I absolutely love waking up at 5am for no reason 🙃" -> predicted **neutral** instead of negative, because the negator "no" accidentally flipped "love," canceling the score to 0.

### Vocabulary gaps cause silent failures

"best day ever honestly" -> predicted **neutral**, true label **positive**.
The diagnose output showed every single token was ignored: `['best', 'day', 'ever', 'honestly']`. None of these common positive words are in `POSITIVE_WORDS`. The model returned neutral with a score of 0 -- not because the text is neutral, but because the model literally could not see any sentiment. The same happened with "this is so frustrating I want to scream" -> **neutral** instead of negative, because "frustrating" and "scream" are not in `NEGATIVE_WORDS`.

### Mixed label only works when both word lists match

"I'm stressed but at least it's almost Friday" -> predicted **negative**, true label **mixed**.
Only "stressed" matched (negative list). The hopeful tone of "at least it's almost Friday" has no matching positive word, so the model cannot trigger the "mixed" label. The mixed detection requires at least one word from each list to be present -- single-sided ambiguity always defaults to positive, negative, or neutral.

### The ML model memorizes rather than generalizes

The ML model achieved 100% accuracy, but it was evaluated on the same 27 posts it trained on. It memorized the answers rather than learning generalizable patterns. With only 27 training examples, it likely learned spurious correlations -- for example, it may have learned that "5am" means negative simply because it appeared in one sarcastic post. On truly new text, its performance would drop significantly.

### Emojis with context-dependent meaning

"I'm fine 🙂" -> predicted **neutral** by the rule based model. The emoji 🙂 is often used passive-aggressively online (meaning "I'm NOT fine"), but it could also be genuine. The model has no way to distinguish between these uses. Similarly, 💀 can mean "dying laughing" (positive) or be used literally in a negative context.

### Short or ambiguous messages

"meh" and "This is fine" both scored 0 and returned neutral. While "meh" is arguably neutral, "This is fine" is often used sarcastically online. Neither model can reliably interpret these without more context.

## 7. Ethical Considerations

### Bias and scope

**Who this model is optimized for:** The dataset reflects one person's language -- a young, English-speaking, internet-literate user familiar with Gen Z slang ("no cap", "lowkey", "ngl", "sick") and emoji culture. The word lists and labels are tuned for this specific communication style.

**Who it might misinterpret:**
- **Non-native English speakers** -- someone writing "I am having the good day" uses correct sentiment words but unusual phrasing that could interact unpredictably with the negation window.
- **Older generations or formal writers** -- someone writing "I find this rather disagreeable" would get neutral because "disagreeable" is not in the negative list, while the model is tuned for casual/informal text.
- **Regional dialects and slang** -- expressions from different English-speaking communities may use unfamiliar vocabulary or phrases that the word lists don't cover.
- **Different cultural contexts** -- sarcasm, politeness norms, and emotional expression vary across cultures. "This is fine" reads as sarcastic in American internet culture but could be genuinely neutral in other contexts.
- **Users expressing distress indirectly** -- "I'm fine 🙂" or "it's whatever I guess" could mask real negative feelings. If this model were used in a mental health application, these false neutrals could mean missed warning signs.

All 27 labels were assigned by a single person, embedding that individual's interpretation of tone and mood. What one person labels "mixed," another might call "negative." This subjectivity is baked into both models -- the rule based model through the word lists, and the ML model through the training labels.

## 8. Ideas for Improvement

- Add more labeled data from diverse sources and language styles
- Use TF-IDF instead of CountVectorizer to weight rare but meaningful words higher
- Split data into training and test sets to measure real generalization (not just training accuracy)
- Add better preprocessing for emojis -- map them to sentiment words before scoring
- Implement phrase-level matching ("falling apart", "no cap") instead of single-word matching
- Use a pre-trained language model (like a small transformer) that understands context and sarcasm
- Add weighted scoring -- give stronger words like "hate" or "amazing" higher weights than mild words like "good" or "bad"
- Collect labels from multiple annotators to reduce individual bias
- Add a confidence score alongside the label so users know when the model is uncertain
