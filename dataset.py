"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    "unreal",
    "obsessed",
    "proud",
    "sick",
    "wicked",
    "fire",
    "lit",
    "dope",
    "goated",
    "immaculate",
    "beautiful",
    "wonderful",
    "fantastic",
    "lovely",
    "hopeful",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    "rough",
    "complicated",
    "exhausted",
    "frustrated",
    "annoyed",
    "miserable",
    "stuck",
    "depressed",
    "anxious",
    "overwhelmed",
    "disappointed",
    "falling",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    
    # Add 5-10 more posts and labels.
    "ngl this week has been rough 😭",
    "lowkey obsessed with this song rn 🔥",
    "I absolutely love waking up at 5am for no reason 🙃",
    "just submitted my project idk if it's good but it's DONE",
    "meh",
    "everything's falling apart but at least I have coffee ☕",
    "no cap that sunset was unreal 💀🧡",
    "why does everything have to be so complicated",
    "I love getting stuck in traffic",
    "That exam was sick",
    "Not bad actually",
    "I'm exhausted but proud of myself",
    "I hate how much I love this song",
    "The food was bad but the vibes were immaculate",
    "I don't think this is good",
    "best day ever honestly",
    "I can't stop smiling 😊",
    "this is so frustrating I want to scream",
    "it's whatever I guess",
    "the vibes are immaculate today",
    "I'm stressed but at least it's almost Friday",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    
    # Add 5-10 more posts and labels.
    "negative",  # "ngl this week has been rough 😭"
    "positive",  # "lowkey obsessed with this song rn 🔥"
    "negative",  # "I absolutely love waking up at 5am for no reason 🙃" (sarcasm)
    "mixed",     # "just submitted my project idk if it's good but it's DONE"
    "neutral",   # "meh"
    "mixed",     # "everything's falling apart but at least I have coffee ☕"
    "positive",  # "no cap that sunset was unreal 💀🧡"
    "negative",  # "why does everything have to be so complicated"
    "negative",  # "I love getting stuck in traffic" (sarcasm)
    "positive",  # "That exam was sick" (slang)
    "positive",  # "Not bad actually" (negation)
    "mixed",     # "I'm exhausted but proud of myself"
    "mixed",     # "I hate how much I love this song"
    "mixed",     # "The food was bad but the vibes were immaculate"
    "negative",  # "I don't think this is good" (wide negation)
    "positive",  # "best day ever honestly"
    "positive",  # "I can't stop smiling 😊"
    "negative",  # "this is so frustrating I want to scream"
    "neutral",   # "it's whatever I guess"
    "positive",  # "the vibes are immaculate today"
    "mixed",     # "I'm stressed but at least it's almost Friday"
]


#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
